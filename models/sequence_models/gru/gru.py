
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1, bidirectional=True):

        """
            Initialization of GRU instance
        Args:
            n_in: int, number of input
            n_hidden: int, number of hidden layers
            dropout: flat, dropout
            num_layers: int, number of layers
            bidirectional: bool, bidirectional
        """

        super(GRU, self).__init__()
        self.gru = nn.GRU(
            n_in,
            n_hidden,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
            num_layers=num_layers,
        )

    def forward(self, input_feat):
        recurrent, _ = self.gru(input_feat)
        return recurrent