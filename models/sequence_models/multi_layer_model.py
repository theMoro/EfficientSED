
import torch
import torch.nn as nn


class MultiLayerModel(nn.Module):
    def __init__(self, layers, skip_connection=True):
        super(MultiLayerModel, self).__init__()
        self.layers = layers
        self.skip_connection = skip_connection

    def forward(self, x):
        for seq_model_layer in self.layers:
            if self.skip_connection:
                x = seq_model_layer(x) + x
            else:
                x = seq_model_layer(x)
        return x


class MultiLayerModelFp32(nn.Module):
    def __init__(self, layers, skip_connection=True):
        super(MultiLayerModelFp32, self).__init__()
        self.layers = layers
        self.skip_connection = skip_connection

    def forward(self, x):
        if self.training:
            with torch.autocast(device_type="cuda", enabled=False):
                x = x.to(torch.float32)

                for seq_model_layer in self.layers:
                    if self.skip_connection:
                        x = seq_model_layer(x) + x
                    else:
                        x = seq_model_layer(x)
        else:
            for seq_model_layer in self.layers:
                if self.skip_connection:
                    x = seq_model_layer(x) + x
                else:
                    x = seq_model_layer(x)

        return x