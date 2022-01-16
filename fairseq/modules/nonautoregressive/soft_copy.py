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

        indicies_abs_diff = ( torch.arange(0, max_seq_len).view(-1, 1).repeat(1, max_seq_len) - torch.arange(0, max_seq_len).view(1, -1).repeat(max_seq_len, 1) ).abs()

        self.register_buffer("indicies_abs_diff", indicies_abs_diff)

        self.softmax = nn.Softmax(dim=-1)

        self.tau = nn.Parameter(torch.tensor(1.))

        return

    def forward(self, embeddings, target_len=None):
        """
        param: embeddings ~ [ sequence_len, batch_size, embedding_dim ] ( [ batch_size, sequence_len, embedding_dim ] if batch_first=True )
        param: target_len ~ (int, None) --- length of target sequence. If None, source_sequene_len will be assumed
        """


        if not self.batch_first:
            embeddings = embeddings.transpose(0, 1)

        source_len = embeddings.size(1)
        if target_len is None:
            target_len = source_len

        assert source_len <= self.max_seq_len, "SoftCopy got embeddings greater then max_seq_len"
        assert target_len <= self.max_seq_len, "SoftCopy source_len greater then max_seq_len"

        # [ target_len, source_len ]
        indicies_diff = self.indicies_abs_diff[:target_len, :source_len]

        print("indicies_diff", indicies_diff.shape)
        print("embeddings", embeddings.shape)
        print("sm", self.softmax(indicies_diff / self.tau).shape)

        # [ batch_size, target_len, embeddings ]
        target_embeddings = self.softmax(indicies_diff / self.tau) @ embeddings

        if not self.batch_first:
            target_embeddings = target_embeddings.transpose(0, 1)

        return target_embeddings
