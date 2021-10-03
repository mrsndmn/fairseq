import torch
import torch.nn as nn

import math

from .hard_concrete_gate import HardConcreteGate

class HCGAttention(nn.Module):
    def __init__(self, hidden_dim: int, key_and_query_dim: int = 64, value_dim: int = 64):
        super(HCGAttention, self).__init__()

        self.query_weights = nn.Linear(hidden_dim, key_and_query_dim)
        self.key_weights = nn.Linear(hidden_dim, key_and_query_dim)
        self.value_weights = nn.Linear(hidden_dim, value_dim)

        self.kq_dim_root = math.sqrt(key_and_query_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.neg_inf = -1e9

        self.attention: torch.Tensor = None

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
        scaled_kv = torch.matmul(queries, keys_transposed) / self.kq_dim_root # # bs, q_seq_len, k_seq_len
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

        result = torch.matmul(scaled_kv, values) # bs, q_seq_len, v_dim

        return result.permute((1, 0, 2))


class MultiHeadHCGAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads=8, key_and_query_dim: int = None, value_dim: int = None, with_hard_concrete_gate=False, hcg_l0_penalty_lambda=0.0):
        super(MultiHeadHCGAttention, self).__init__()

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
