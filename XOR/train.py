import torch
from Model import XORModel

def trainXORModel(model: XORModel, dataset: tuple[torch.Tensor, torch.Tensor], epochs=2000, learning_rate=0.1, device="cpu"):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.to(device)
    criterion.to(device)

    print(f"Using {device} device\nLearning Rata {learning_rate}\nEpochs {epochs}")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(dataset[0].to(device))

        loss = criterion(outputs, dataset[1].to(device))
        loss.backward()

        optimizer.step()

        if(epoch % 200 == 0):
            print(f"Epoch {epoch}, Loss: {loss.item()}")