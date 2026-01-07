import torch
from Model import ANDModel
from Data import dataset
from train import trainANDModel

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

model = ANDModel()

model.load_state_dict(torch.load("model_weights.pth"))
model.eval()  # coloca em modo de inferência

trainANDModel(model=model, dataset=dataset)
torch.save(model.state_dict(), "model_weights.pth")

with torch.no_grad():
    model.to(device)
    pred = model(dataset[0].to(device))
    print(f"Predict: {pred}\nResult: {(pred > 0.5).float()}")