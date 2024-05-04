import torch.nn as nn

class FeatureTransferNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        layers=[]
        for idx in range(len(hidden_dim)):
            in_ch = input_dim if idx == 0 else hidden_dim[idx - 1]
            out_ch = hidden_dim[idx]
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
