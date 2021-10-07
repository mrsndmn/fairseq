import torch
from torch import Tensor, nn

import math

from .hard_concrete_gate import HardConcreteGate
from fairseq.modules import MultiheadAttention
from typing import Dict, Optional, Tuple


import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter


class MultiHeadHCGAttention(MultiheadAttention):
    def __init__(self, *args, **kwargs):

        hcg_log_a = kwargs.pop("hcg_log_a", 0.0)
        hcg_temperature = kwargs.pop("hcg_temperature", 0.5)
        hcg_adjust_range = kwargs.pop("hcg_adjust_range", (-0.1, 1.1))

        self.with_hard_concrete_gate = kwargs.pop("with_hard_concrete_gate", False)

        super().__init__(*args, **kwargs)

        if self.with_hard_concrete_gate:
            self.hcg = HardConcreteGate(self.num_heads,
                                        log_a=hcg_log_a,
                                        temperature=hcg_temperature,
                                        adjust_range=hcg_adjust_range,)

    @classmethod
    def from_fairseq_mha(cls, fairseq_mha: MultiheadAttention, **kwargs):
        hcg_mha = cls(fairseq_mha.embed_dim, fairseq_mha.num_heads,
                        kdim=fairseq_mha.kdim, vdim=fairseq_mha.vdim, dropout=fairseq_mha.dropout_module.p,
                        self_attention=fairseq_mha.self_attention, encoder_decoder_attention=fairseq_mha.encoder_decoder_attention,
                        add_zero_attn=fairseq_mha.add_zero_attn, **kwargs)

        hcg_mha.k_proj = fairseq_mha.k_proj
        hcg_mha.q_proj = fairseq_mha.q_proj
        hcg_mha.v_proj = fairseq_mha.v_proj

        hcg_mha.out_proj = fairseq_mha.out_proj

        hcg_mha.bias_k = fairseq_mha.bias_k
        hcg_mha.bias_v = fairseq_mha.bias_v

        return hcg_mha

    def setup_hcg(self, hcg_log_a=0., hcg_temperature=0.5, hcg_adjust_range=(-0.1, 1.1)):
        self.with_hard_concrete_gate = True
        self.hcg = HardConcreteGate(self.num_heads,
                                    log_a=hcg_log_a,
                                    temperature=hcg_temperature,
                                    adjust_range=hcg_adjust_range,)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        # attn ~ [ tgt_len, bsz, embed_dim ]
        if self.with_hard_concrete_gate:
            attn = self.hcg(attn)

        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


class HCGAttention(nn.Module):
    def __init__(self, hidden_dim: int, key_and_query_dim: int = 64, value_dim: int = 64):
        super(HCGAttention, self).__init__()

        self.query_weights = nn.Linear(hidden_dim, key_and_query_dim)
        self.key_weights = nn.Linear(hidden_dim, key_and_query_dim)
        self.value_weights = nn.Linear(hidden_dim, value_dim)

        self.kq_dim_inv_root = 1 / math.sqrt(key_and_query_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.neg_inf = -1e9

        self.attention: torch.Tensor = None

        nn.init.xavier_uniform_(self.query_weights.weight , gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.key_weights.weight , gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.value_weights.weight , gain=1 / math.sqrt(2))

    def forward(self, q_hidden_inputs: torch.Tensor, k_hidden_inputs: torch.Tensor, v_hidden_inputs: torch.Tensor, attn_mask=None, key_padding_mask=None, save_attention=False):
        """
        {kqv}_hidden_inputs: batch_size, seq_len, model_hid_dim
        """


        q_hidden_inputs = q_hidden_inputs.permute((1, 0, 2))
        k_hidden_inputs = k_hidden_inputs.permute((1, 0, 2))
        v_hidden_inputs = v_hidden_inputs.permute((1, 0, 2))

        if key_padding_mask is not None:
            key_padding_mask = torch.repeat_interleave(torch.unsqueeze(key_padding_mask, -1), k_hidden_inputs.shape[-1], dim=-1)
            # print("key_padding_mask.shape", key_padding_mask.shape)
            # print("k_hidden_inputs.shape", k_hidden_inputs.shape)
            k_hidden_inputs = k_hidden_inputs.masked_fill(key_padding_mask, self.neg_inf)

        queries: torch.Tensor = self.query_weights(q_hidden_inputs) # bs, q_seq_len, q_dim
        keys: torch.Tensor = self.key_weights(k_hidden_inputs)      # bs, k_seq_len, k_dim
        values: torch.Tensor = self.value_weights(v_hidden_inputs)  # bs, v_seq_len, v_dim

        batch_size = queries.size(0)
        q_seq_len = queries.size(1)
        k_seq_len = keys.size(1)
        v_seq_len = values.size(1)
        # print("q_seq_len", q_seq_len)
        # print("k_seq_len", k_seq_len)
        # print("v_seq_len", v_seq_len)
        assert k_seq_len == v_seq_len, f'k_seq_len{k_seq_len} == v_seq_len{v_seq_len}'


        keys_transposed = keys.permute(0, 2, 1) # bs, k_dim, k_seq_len

        # print("queries.shape", queries.shape, "keys_transposed.shape", keys_transposed.shape)
        scaled_kv = torch.bmm(queries, keys_transposed) * self.kq_dim_inv_root # # bs, q_seq_len, k_seq_len
        assert scaled_kv.size() == torch.Size((batch_size, q_seq_len, k_seq_len))

        if attn_mask is not None:
            attn_mask = torch.unsqueeze(attn_mask.permute((1, 0)), -1)
            # print("scaled_kv.size()", scaled_kv.size())
            # print("mask.size()", attn_mask.size())
            # scaled_kv.masked_fill_(mask == False, self.neg_inf)

            # todo какая тут маска? правильно ли знак стоит?
            # должны ли маски для паддингов и для аттеншна одновременно накладываться?
            # scaled_kv.masked_fill_(mask.permute((1, 0)), self.neg_inf)
            scaled_kv.masked_fill_(attn_mask, self.neg_inf)


        scaled_kv = self.softmax(scaled_kv)

        if save_attention:
            self.attention = scaled_kv

        result = torch.bmm(scaled_kv, values) # bs, q_seq_len, v_dim

        return result.permute((1, 0, 2))


class MultiHeadHCGAttention_BACKUP(nn.Module):
    def __init__(self, hidden_dim: int, num_heads=8, key_and_query_dim: int = None, value_dim: int = None, with_hard_concrete_gate=False, hcg_l0_penalty_lambda=0.0):
        super(MultiHeadHCGAttention_BACKUP, self).__init__()

        if key_and_query_dim is None:
            key_and_query_dim = hidden_dim // num_heads
        if value_dim is None:
            value_dim = key_and_query_dim

        if hidden_dim // num_heads != key_and_query_dim:
            raise ValueError(f"hidden_dim must be equal to num_heads * key_and_query_dim. Got: hidden_dim={hidden_dim} // num_heads={num_heads} != key_and_query_dim={key_and_query_dim}")

        attentions = []
        concrete_gates = []
        for _ in range(num_heads):
            attentions.append(HCGAttention(hidden_dim, key_and_query_dim=key_and_query_dim, value_dim=value_dim))
            if with_hard_concrete_gate:
                concrete_gates.append( HardConcreteGate(1, l0_penalty_lambda=hcg_l0_penalty_lambda) )

        self.attention_heads = nn.ModuleList(attentions)

        self.hard_concrete_gates = nn.ModuleList(concrete_gates)

        self.heads_weights = nn.Linear(num_heads * value_dim, hidden_dim)

    @classmethod
    def from_fairseq_mha(cls, fairseq_mha: MultiheadAttention, with_hard_concrete_gate=False, hcg_l0_penalty_lambda=0.0):


        if not fairseq_mha.qkv_same_dim:
            raise ValueError("self.qkv_same_dim == False is not supported")

        hcg_mha = cls(hidden_dim=fairseq_mha.embed_dim, num_heads=fairseq_mha.num_heads, with_hard_concrete_gate=with_hard_concrete_gate, hcg_l0_penalty_lambda=0.0)

        head_dim = fairseq_mha.head_dim
        for i, attention_head in enumerate(hcg_mha.attention_heads):
            pass

        return hcg_mha

    # todo костыль!
    def _get_input_buffer(self, *args, **kwargs):
        return None

    @property
    def num_heads(self):
        return len(self.attention_heads)

    HAS_UNKNOWN_ARGS = False

    def forward(self, query=None,
                key=None,
                value=None,
                key_padding_mask=None,
                need_weights=False,
                attn_mask=None,
                # self_attn_mask=None,
                self_attn_padding_mask=None,
                **kwargs):
        if len(kwargs) > 0 and not self.HAS_UNKNOWN_ARGS:
            self.HAS_UNKNOWN_ARGS = True
            print("unknown args:", kwargs)


        q_hidden_inputs: torch.Tensor = query
        k_hidden_inputs: torch.Tensor = key
        v_hidden_inputs: torch.Tensor = value

        # mask = attn_mask
        # if mask is None:
        #     mask = self_attn_padding_mask

        # if key_padding_mask is not None:
        #     if mask is not None:
        #         mask = mask & key_padding_mask
        #     else:
        #         # print("key_padding_mask", key_padding_mask.shape)
        #         mask = key_padding_mask # .permute((1, 0))
        #         # что тут произошло?
        #         mask = torch.repeat_interleave(torch.unsqueeze(mask, -1), mask.shape[1], dim=-1)

        attention_outputs = []
        for attention in self.attention_heads:
            # print("q", q_hidden_inputs.shape, "k", k_hidden_inputs.shape, "v", v_hidden_inputs.shape)
            attention_output = attention(q_hidden_inputs, k_hidden_inputs, v_hidden_inputs, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            attention_outputs.append(attention_output)

        if len(self.hard_concrete_gates) > 0:
            for i, hcg in enumerate(self.hard_concrete_gates):
                attention_outputs[i] = hcg(attention_outputs[i])

        # bs, seq_len, v_dim * num_heads
        attention_outputs = torch.cat(attention_outputs, dim=-1)

        # todo в такой ли же последоватльности возвращает fairseq?
        return self.heads_weights(attention_outputs), None # seq_len, bs, hidd_dim


class SimpleMultiHeadHCGAttention(MultiHeadHCGAttention):
    '''
    The same as MultiHeadHCGAttention but all the query key and value inputs are the same
    '''

    def __init__(self, hidden_dim: int, key_query_value_dim: int = 64, num_heads=8, with_hard_concrete_gate=False):
        super(SimpleMultiHeadHCGAttention, self).__init__(hidden_dim, key_and_query_dim=key_query_value_dim, value_dim=key_query_value_dim, num_heads=num_heads, with_hard_concrete_gate=with_hard_concrete_gate)
        return

    def forward(self, hidden_inputs, mask=None):
        return super(SimpleMultiHeadHCGAttention, self).forward(q_hidden_inputs=hidden_inputs, k_hidden_inputs=hidden_inputs, v_hidden_inputs=hidden_inputs, mask=mask)
