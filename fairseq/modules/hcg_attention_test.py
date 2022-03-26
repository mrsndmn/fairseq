from importlib.metadata import requires
import pytest

import time

from .hcg_attention import HCGAttention, MultiHeadHCGAttention, SimpleMultiHeadHCGAttention

import torch
from fairseq.modules import MultiheadAttention

from .hard_concrete_gate import HardConcreteGate

from fairseq import checkpoint_utils, options, tasks, utils

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_from_fairseq_mha():
    # todo test id and disable
    # self.dropout_module
    # todo test it
    # self.scaling

    models, model_dict = checkpoint_utils.load_model_ensemble(utils.split_paths( 'checkpoints_nonautoregressive_transformer_orig/checkpoint_best.pt' ))
    model = models[0]
    # print(model)

    mha_self_attention: MultiheadAttention = model.encoder.layers[0].self_attn

    assert mha_self_attention.self_attention
    assert not mha_self_attention.encoder_decoder_attention

    mha_self_attention.dropout_module.p = 0

    hcg_mha = MultiHeadHCGAttention.from_fairseq_mha(mha_self_attention, with_hard_concrete_gate=False)

    print("count_trainable_params", count_trainable_params(hcg_mha))
    print("hcg_mha", hcg_mha)
    print("mha_self_attention", mha_self_attention)
    # assert count_trainable_params(hcg_mha) == count_trainable_params(mha_self_attention)

    fairseq_modules_gen = list(mha_self_attention.modules())[1:]
    hcg_modules_gen = list(x for x in mha_self_attention.modules() if type(x) != HardConcreteGate)[1:]

    for hcg_m, gairseq_m in zip(fairseq_modules_gen, hcg_modules_gen):
        for hcg_p, gairseq_p in zip(hcg_m.parameters(), gairseq_m.parameters()):
            assert (hcg_p == gairseq_p).all()

    rand_t = torch.rand( (7, 3, 512) )
    hcg_result, _ = hcg_mha.forward(query=rand_t, key=rand_t, value=rand_t)
    mha_result, _ = mha_self_attention.forward(query=rand_t, key=rand_t, value=rand_t)

    assert hcg_result.size() == mha_result.size()



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

def _count_trainable_params(model):
    return sum( p.numel() for p in model.parameters() if p.requires_grad )

def test_prune_hgc_attention():
    with torch.no_grad():
        kq_dim = 4
        v_dim = 8
        num_heads = 16
        hidden_dim = 64

        batch_size = 3
        seq_len = 7
        attention_input = torch.rand((batch_size, seq_len, hidden_dim))

        mha = MultiHeadHCGAttention(hidden_dim, num_heads=num_heads, with_hard_concrete_gate=True)
        mha = mha.eval()

        print("mha.out_proj", mha.out_proj)

        mha.hcg.log_a[0] = float("+inf")
        mha.hcg.log_a[1:] = float("-inf")

        attention, attention_weights = mha.forward(attention_input, key=attention_input, value=attention_input)
        assert attention.size() == attention_input.size()

        print("attention.shape", attention.shape)
        print("before pruning:", _count_trainable_params(mha))

        mha.prune()

        print(mha.v_proj)
        print("mha.v_proj.weight.shape", mha.v_proj.weight.shape)
        print("mha.v_proj.bias.shape", mha.v_proj.bias.shape)

        print("after pruning:", _count_trainable_params(mha))

        attention_pruned, attention_weights = mha.forward(attention_input, key=attention_input, value=attention_input)
        assert attention.size() == attention_input.size()

        print(attention_pruned[0, 0, :10])
        print(attention[0, 0, :10])

        assert (attention_pruned == attention).all()

def test_speedup_pruned_mha():
    with torch.no_grad():
        kq_dim = 4
        v_dim = 8
        num_heads = 16
        hidden_dim = 64

        batch_size = 3
        seq_len = 7
        attention_input = torch.rand((batch_size, seq_len, hidden_dim))

        mha = MultiHeadHCGAttention(hidden_dim, num_heads=num_heads, with_hard_concrete_gate=True)
        mha = mha.eval()

        print("mha.out_proj", mha.out_proj)

        mha.hcg.log_a[0] = float("+inf")
        mha.hcg.log_a[1:] = float("-inf")

        timing_iterations = 10000

        start_not_pruned = time.time()
        for _ in range(timing_iterations):
            attention, attention_weights = mha.forward(attention_input, key=attention_input, value=attention_input)
        not_pruned_duration = time.time() - start_not_pruned

        assert attention.size() == attention_input.size()

        print("attention.shape", attention.shape)
        print("before pruning:", _count_trainable_params(mha))

        mha.prune()

        print(mha.v_proj)
        print("mha.v_proj.weight.shape", mha.v_proj.weight.shape)
        print("mha.v_proj.bias.shape", mha.v_proj.bias.shape)

        print("after pruning:", _count_trainable_params(mha))

        start_pruned = time.time()
        for _ in range(timing_iterations):
            attention_pruned, attention_weights = mha.forward(attention_input, key=attention_input, value=attention_input)
        pruned_duration = time.time() - start_pruned

        print("pruned_duration", pruned_duration)
        print("not pruned_duration", not_pruned_duration)

        assert attention.size() == attention_input.size()

        print(attention_pruned[0, 0, :10])
        print(attention[0, 0, :10])

        assert (attention_pruned == attention).all()
