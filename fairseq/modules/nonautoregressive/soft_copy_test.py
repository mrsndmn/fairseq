import pytest
import torch
from fairseq.modules.nonautoregressive.soft_copy import SoftCopy

def test_soft_copy_shapes():

    sc = SoftCopy()

    src_embeddings = torch.rand( [17, 8, 256] )
    trg_embeddings = sc(src_embeddings)

    assert trg_embeddings.size() == src_embeddings.size()

def test_soft_copy_different_target_shape():
    sc = SoftCopy()

    src_embeddings = torch.rand( [17, 8, 256] )
    target_len = src_embeddings.size(0) * 3
    trg_embeddings = sc(src_embeddings, target_len=target_len)

    assert trg_embeddings.size(0) == target_len
    assert trg_embeddings.size(1) == src_embeddings.size(1)
    assert trg_embeddings.size(2) == src_embeddings.size(2)
