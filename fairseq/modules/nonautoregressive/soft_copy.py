import torch
import torch.nn as nn

class SoftCopy(nn.Module):
    # todo requiers weights initialization

    def __init__(self, batch_first=False, max_seq_len=512):
        """

        https://arxiv.org/pdf/1906.02041.pdf

        """

        super().__init__()

        self.batch_first = batch_first
        self.max_seq_len = max_seq_len

        indicies_abs_diff = ( torch.arange(0, 10).view(-1, 1).repeat(1, 10) - torch.arange(0, 10).view(1, -1).repeat(10, 1) ).abs()

        self.register_buffer("indicies_abs_diff", indicies_abs_diff)

        self.softmax = nn.Softmax(dim=-1)

        self.tau = nn.Parameter(torch.tensor(1))

        return

    def forward(self, embeddings, target_len=None):
        """
        param: embeddings ~ [ sequence_len, batch_size, embedding_dim ] ( [ batch_size, sequence_len, embedding_dim ] if batch_first=True )
        param: target_len ~ (int, None) --- length of target sequence. If None, source_sequene_len will be assumed
        """


        if self.batch_first:
            embeddings = embeddings.transpose(0, 1)

        source_len = embeddings.size(0)
        if target_len is None:
            target_len = source_len

        assert source_len <= self.max_seq_len, "SoftCopy got embeddings greater then max_seq_len"
        assert target_len <= self.max_seq_len, "SoftCopy source_len greater then max_seq_len"

        # [ target_len, source_len ]
        indicies_diff = self.indicies_abs_diff[:target_len, :source_len]

        # [ target_len, batch_size, embeddings ]
        target_embeddings = self.softmax(indicies_diff / self.tau) @ embeddings

        return target_embeddings
