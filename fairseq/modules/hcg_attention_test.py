import pytest

from .hcg_attention import HCGAttention, MultiHeadHCGAttention, SimpleMultiHeadHCGAttention, AddAndNorm

import torch

def test_attention_forward():
    with torch.no_grad():
        kq_dim = 2
        v_dim = 2
        hidden_dim = 16
        att = HCGAttention(hidden_dim, key_and_query_dim=kq_dim, value_dim=v_dim)

        batch_size = 4
        q_seq_len = 2
        kv_seq_len = 7

        q_attention_input = torch.rand((batch_size, q_seq_len, hidden_dim))
        kv_attention_input = torch.rand((batch_size, kv_seq_len, hidden_dim))


        attention_output = att.forward(q_hidden_inputs=q_attention_input, k_hidden_inputs=kv_attention_input, v_hidden_inputs=kv_attention_input, mask=None)
        assert attention_output.size() == torch.Size((batch_size, q_seq_len, v_dim))
        assert attention_output.sum().item() != 0

        mask = torch.ones((batch_size, q_seq_len, kv_seq_len))
        mask[:, :, -1] = 0

        attention_output = att.forward(q_hidden_inputs=q_attention_input, k_hidden_inputs=kv_attention_input, v_hidden_inputs=kv_attention_input, mask=mask, save_attention=True)
        assert attention_output.size() == torch.Size((batch_size, q_seq_len, v_dim))
        assert att.attention[:, :, -1].sum().item() == 0


def test_multihead_attention_invalid_args():
    with pytest.raises(ValueError):
        MultiHeadHCGAttention(16, key_and_query_dim=4, value_dim=4, num_heads=5)

    with pytest.raises(ValueError):
        SimpleMultiHeadHCGAttention(16, key_query_value_dim=4, num_heads=5)

def test_multihead_attention():
    with torch.no_grad():
        kq_dim = 4
        v_dim = 8
        num_heads = 16
        hidden_dim = 64

        batch_size = 3
        seq_len = 7
        attention_input = torch.rand((batch_size, seq_len, hidden_dim))

        mha = MultiHeadHCGAttention(hidden_dim, key_and_query_dim=kq_dim, value_dim=v_dim, num_heads=num_heads)
        mha_ouptut = mha.forward(q_hidden_inputs=attention_input, k_hidden_inputs=attention_input, v_hidden_inputs=attention_input, mask=None)
        assert mha_ouptut.size() == attention_input.size()

def test_simple_multi_head_attention():
    with torch.no_grad():
        kqv_dim = 4
        num_heads = 16
        hidden_dim = 64

        batch_size = 3
        seq_len = 7
        attention_input = torch.rand((batch_size, seq_len, hidden_dim))

        smha = SimpleMultiHeadHCGAttention(hidden_dim, key_query_value_dim=kqv_dim, num_heads=num_heads)
        smha_ouptut = smha.forward(attention_input, mask=None)
        assert smha_ouptut.size() == attention_input.size()

        mask = torch.ones((batch_size, seq_len, seq_len))
        mask[:, :, -1] = 0

        smha_ouptut = smha.forward(attention_input, mask=mask)
        assert smha_ouptut.size() == attention_input.size()
