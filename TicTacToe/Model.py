import torch

class TicTacToeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(9, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 9)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)