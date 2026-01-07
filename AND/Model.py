from torch import nn, sigmoid

class ANDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1) # 2 entradas (0 ou 1) -> 1 saida

    def forward(self, x):
        return sigmoid(self.layer(x)) # sigmoid -> sáida entre 0 e 1
    