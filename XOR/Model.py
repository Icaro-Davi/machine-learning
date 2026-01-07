from torch import nn, sigmoid

class XORModel(nn.Module):
    def __init__(self):
        super().__init__();
        self.net = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 1)
        )
    
    def forward(self, input):
        return sigmoid(self.net(input)) # Sigmoid retorna valores entre 0 e 1 que pode interpretar como probabildiades