import torch
from Model import TicTacToeModel
from server import start

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = TicTacToeModel()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

start(model, device)