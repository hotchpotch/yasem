import logging
from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pooling type は max or mean という typehint を追加
POOLING_TYPE = Literal["max", "mean"]


class SpladeSubwordProcessor:
    DEFAULT_MASK_INDEX_ID = -100

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        subword_prefix: str = "##",
        mask_index_id: int = DEFAULT_MASK_INDEX_ID,
        pooling_type: POOLING_TYPE = "mean",
    ):
        self.subword_token_ids = set()
        self.mask_index_id = mask_index_id
        self.pooling_type: POOLING_TYPE = pooling_type

        for token in tokenizer.get_vocab():
            if token.startswith(subword_prefix):
                self.subword_token_ids.add(tokenizer.convert_tokens_to_ids(token))  # type: ignore
        if len(self.subword_token_ids) == 0:
            raise ValueError(
                f"Subword token prefix '{subword_prefix}' not found in tokenizer vocab."
            )

    def create_subword_indices(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        トークンIDからサブワードインデックスを生成する
        サブワードを含む単語は同じインデックスでグループ化し、
        単独トークンはmask_index_idとして扱う

        Args:
            token_ids (torch.Tensor): トークンID (batch_size, seq_len)

        Returns:
            torch.Tensor: サブワードインデックス (batch_size, seq_len)
                mask_index_id: 単独トークン（サブワードを含まない単語）やパディング
                0以上: サブワードを含む単語のグループインデックス
        """
        mask_index_id = self.mask_index_id
        subword_token_ids = self.subword_token_ids
        batch_size, seq_len = token_ids.shape
        subword_indices = torch.full_like(
            token_ids,
            mask_index_id,  # PADDINGのマスク値
        )

        current_subword_group_idx = -1
        for b in range(batch_size):
            word_start_pos = -1
            in_subword_sequence = False

            for i in range(seq_len):
                token_id = token_ids[b, i].item()

                # パディングやマスクされたトークンはスキップ
                if token_id == mask_index_id:
                    continue

                is_subword = token_id in subword_token_ids

                # 新しい単語の開始
                if not is_subword and not in_subword_sequence:
                    # 前の単語の処理
                    if word_start_pos != -1:
                        # 単独トークンの場合
                        if not in_subword_sequence:
                            subword_indices[b, word_start_pos] = mask_index_id

                    word_start_pos = i
                    in_subword_sequence = False

                # サブワードシーケンスの開始
                elif is_subword and not in_subword_sequence:
                    current_subword_group_idx += 1
                    in_subword_sequence = True
                    # 直前のトークンも同じグループに
                    if word_start_pos != -1:
                        subword_indices[b, word_start_pos : i + 1] = (
                            current_subword_group_idx
                        )

                # サブワードシーケンスの途中
                elif is_subword and in_subword_sequence:
                    subword_indices[b, i] = current_subword_group_idx

                # サブワードシーケンスの終了
                if not is_subword and in_subword_sequence:
                    word_start_pos = i
                    in_subword_sequence = False

            # 最後の単語の処理
            if word_start_pos != -1 and not in_subword_sequence:
                subword_indices[b, word_start_pos] = mask_index_id

        return subword_indices

    def aggregate_subwords(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        subword_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimized aggregation of subword logits using PyTorch's scatter operations.

        Args:
            logits (torch.Tensor): [batch_size, vocab_size] logits tensor after splade_max.
            input_ids (torch.Tensor): [batch_size, sequence_length] input token IDs.
            subword_indices (torch.Tensor): [batch_size, sequence_length] subword group indices.

        Returns:
            torch.Tensor: Aggregated logits tensor.
        """
        batch_size, vocab_size = logits.size()
        device = logits.device
        dtype = logits.dtype  # Ensure consistent dtype

        # マスクを作成して有効なサブワード位置を特定
        mask = subword_indices != self.mask_index_id  # [B, S]

        # 有効な位置のインデックスを取得
        valid_positions = mask.nonzero(
            as_tuple=False
        )  # [N, 2] 各行は (batch_idx, seq_idx)
        if valid_positions.numel() == 0:
            return logits  # No valid subwords to aggregate

        batch_indices = valid_positions[:, 0]  # [N]
        seq_indices = valid_positions[:, 1]  # [N]

        # 該当するグループインデックスとトークンIDを抽出
        group_indices = subword_indices[batch_indices, seq_indices]  # [N]
        token_ids = input_ids[batch_indices, seq_indices]  # [N]

        # 選択されたトークンIDに対応するlogitsを取得
        selected_logits = logits[batch_indices, token_ids]  # [N]

        # 各バッチごとにグループIDをオフセットしてユニークなグループ識別子を計算
        max_group_tensor = group_indices.max()
        max_group = (
            max_group_tensor.item()
            if torch.is_tensor(max_group_tensor)
            else max_group_tensor
        )
        if max_group < 0:
            max_group = 0  # Handle case where all group_indices are -100, though unlikely due to mask

        # グループ識別子を一意にするためにバッチオフセットを加算
        group_offset = batch_indices * (max_group + 1)  # [N]
        unique_group_ids = group_offset + group_indices  # [N]

        # ユニークなグループの数を計算
        num_unique_groups = batch_size * (max_group + 1)  # Removed .item()

        if self.pooling_type == "max":
            # -infで初期化し、logitsと同じdtypeを設定
            pooled_values = torch.full(
                (num_unique_groups,),  # type: ignore
                -float("inf"),
                device=device,
                dtype=dtype,  # type: ignore
            )
            # scatter_reduceを使用して各グループの最大値を計算
            pooled_values = pooled_values.scatter_reduce(
                dim=0,
                index=unique_group_ids,
                src=selected_logits,
                reduce="amax",
                include_self=True,
            )
        elif self.pooling_type == "mean":
            # 合計とカウントのテンソルを初期化し、logitsと同じdtypeを設定
            sum_pooled = torch.zeros(num_unique_groups, device=device, dtype=dtype)  # type: ignore
            count_pooled = torch.zeros(num_unique_groups, device=device, dtype=dtype)  # type: ignore
            # 各グループの合計を計算
            sum_pooled = sum_pooled.scatter_add(0, unique_group_ids, selected_logits)
            # 各グループのカウントを計算
            count_pooled = count_pooled.scatter_add(
                0, unique_group_ids, torch.ones_like(selected_logits)
            )
            # 平均値を計算（ゼロ除算を防ぐ）
            pooled_values = sum_pooled / torch.clamp(count_pooled, min=1)

        # プールされた値を各トークンにマッピング
        pooled_values_per_token = pooled_values[unique_group_ids]  # [N] # type: ignore

        # バッチ内の複数のトークンを処理するためにグローバルなトークンIDを計算
        global_token_ids = batch_indices * vocab_size + token_ids  # [N]

        # 各トークンごとの最大プール値を保持するテンソルを初期化（-infで初期化）
        final_pooled = torch.full(
            (batch_size * vocab_size,), -float("inf"), device=device, dtype=dtype
        )

        if self.pooling_type == "max":
            # MaxPoolingの場合は最大値を使用
            final_pooled = final_pooled.scatter_reduce(
                dim=0,
                index=global_token_ids,
                src=pooled_values_per_token,
                reduce="amax",
                include_self=True,
            )
            final_pooled = final_pooled.view(batch_size, vocab_size)
            # 元のlogitsと最大値を取る
            new_logits = torch.maximum(logits, final_pooled)
        elif self.pooling_type == "mean":
            # MeanPoolingの場合は平均値を直接使用
            temp_sum = torch.zeros_like(final_pooled)
            temp_count = torch.zeros_like(final_pooled)

            # 値の合計とカウントを集計
            temp_sum.scatter_add_(0, global_token_ids, pooled_values_per_token)
            temp_count.scatter_add_(
                0, global_token_ids, torch.ones_like(pooled_values_per_token)
            )

            # 最終的な平均を計算
            final_pooled = (temp_sum / temp_count.clamp(min=1.0)).view(
                batch_size, vocab_size
            )

            # サブワードがある位置のみ平均値で更新
            subword_mask = (temp_count > 0).view(batch_size, vocab_size)
            new_logits = torch.where(subword_mask, final_pooled, logits)

        return new_logits
