import torch
from Model import ANDModel

def trainANDModel(model: ANDModel, dataset: tuple[torch.Tensor, torch.Tensor], epochs=2000, learning_rate=0.1, device="cpu"):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.to(device)
    criterion.to(device)

    print(f"Using {device} device\nLearning Rata {learning_rate}\nEpochs {epochs}")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(dataset[0])

        loss = criterion(outputs, dataset[1])
        loss.backward()

        optimizer.step()

        if(epoch % 200 == 0):
            print(f"Epoch {epoch}, Loss: {loss.item()}")


# gradient accumulator
# accumulation_steps = 4  # queremos simular batch_size=4

# for epoch in range(50):
#     optimizer.zero_grad()

#     running_loss = 0.0

#     for i in range(len(X)):     # processa 1 por vez
#         x = X[i].unsqueeze(0)   # adiciona batch dimension (1,2)
#         target = y[i].unsqueeze(0)

#         output = model(x)
#         loss = criterion(output, target)

#         # backward acumula gradiente
#         loss.backward()
#         running_loss += loss.item()

#         # quando acumula 4 amostras → atualiza pesos
#         if (i + 1) % accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()

#     print(f"Epoch {epoch}, Loss: {running_loss:.4f}")