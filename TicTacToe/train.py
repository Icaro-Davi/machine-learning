import torch
import random
from typing import cast
from Model import TicTacToeModel
from evaluate import evaluate

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Running on {device}")

playerModel = TicTacToeModel().to(device)
#playerModel.load_state_dict(torch.load('model_weights.pth')) # descomentar para carregar um modelo salvo
playerModel.train()
# evaluate(playerModel, device=device)

enemyModel = TicTacToeModel().to(device)
enemyModel.load_state_dict(playerModel.state_dict())
enemyModel.eval()

gamma = 0.95
epochs = 16000
epsilon = 1.0
minEpsilon = 0
epsilonDecay = 0.9998
matchPenalityDecay = -0.01
learningRate = 1e-3

lossFn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(playerModel.parameters(), lr=learningRate)

class Utilities():
    wins = [
        [0,1,2], [3,4,5], [6,7,8],  # linhas
        [0,3,6], [1,4,7], [2,5,8],  # colunas
        [0,4,8], [2,4,6]            # diagonais
    ]

    def checkWin(board: torch.Tensor): 
        for a,b,c in Utilities.wins:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a].item() # 1 ou -1
        if torch.all(board != 0):
            return 2 # empate
        return 0 # continua

    def valid_moves(board: torch.Tensor):
        return [i for i in range(9) if board[i] == 0]
    
    def hasWinningMove(board: torch.Tensor, player: int):
        for move in Utilities.valid_moves(board):
            tempBoard = board.clone()
            tempBoard[move] = player
            if(Utilities.checkWin(tempBoard) == player): return True
        return False
    
    def isPlayerWin(player: int, matchResult: int):
        return player == matchResult
    
    def isDraw(matchResult: int):
        return matchResult == 2
    
total_loss = 0
toggle_first = False

print(
    f"gamma {gamma} | "
    f"epochs {epochs} | "
    f"epsilon {epsilon} | "
    f"minEpsilon {minEpsilon} | "
    f"epsilonDecay {epsilonDecay} | "
    f"learningRate {learningRate}"
)

# inicia o treinamento
for epoch in range(epochs):
    player = 1
    enemy = -player
    board = torch.zeros(9, device=device)
    matchPenality = 0
    # Muda quem irá iniciar primeiro na partida
    if(toggle_first):
        firstMove = random.choice(Utilities.valid_moves(board))
        board[firstMove] = enemy
        toggle_first = False
    else:
        toggle_first = True

    done = False

    # Iniciar a partida player x enemy
    while not done:
        validMoves = Utilities.valid_moves(board)
        if(len(validMoves) > 0):
            # Joga como o player
            playerBoardState = board.clone() * player # cria a visão do tabuleiro para o player
            # Escolhe movimento do modelo player
            if(random.random() < epsilon):
                playerMove = random.choice(validMoves)
            else:
                with torch.no_grad():
                    playerMovePossibilities = cast(torch.Tensor, playerModel(playerBoardState.unsqueeze(0))[0])
                    playerMovePossibilities[[i for i in range(9) if i not in validMoves]] = -1e9 # sobrescreve posições que já tem jogadas com valor negativo para remover possibilidade de escolha
                    playerMove = torch.argmax(playerMovePossibilities).item() # pega o index do maior valor dentro da lista
            
            board[playerMove] = player # player faz sua jogada

        reward = 0
        matchResult = Utilities.checkWin(board) # 1 player ganha, -1 enemy ganha, 2 empate
        if(Utilities.isPlayerWin(player, matchResult)):
            reward = 1
            done = True        
        elif(Utilities.isDraw(matchResult)): # Pequena recompensa por forçar empate
            reward = 0
            done = True

        validMoves = Utilities.valid_moves(board)
        if(len(validMoves) > 0 and not done):
            # Joga como inimigo
            enemyBoardState = board.clone() * enemy # cria a visão do tabuleiro do inimigo
            if(random.random() < .1):
                enemyMove = random.choice(validMoves)
            else:
                with torch.no_grad():
                    enemyMovePossibilities = cast(torch.Tensor, enemyModel(enemyBoardState.unsqueeze(0)))[0]
                    enemyMovePossibilities[[i for i in range(9) if i not in validMoves]] = -1e9
                    enemyMove = torch.argmax(enemyMovePossibilities).item()
            
            board[enemyMove] = enemy
        
            matchResult = Utilities.checkWin(board) # 1 player ganha, -1 enemy ganha, 2 empate
            if(Utilities.isPlayerWin(enemy, matchResult)): # perdeu é penalizado
                reward = -1
                done = True
            elif(Utilities.isDraw(matchResult)):
                reward = 0.5
                done = True
        
        if(not done):
            reward = matchPenality
            if(len(Utilities.valid_moves(board)) < 7):
                matchPenality += matchPenalityDecay
            if(Utilities.hasWinningMove(playerBoardState, player)): # se ele perdeu o movimento vencedor é penalizado
                reward -= 0.5
            if(Utilities.hasWinningMove(board, enemy)): # se ele deixou um movimento inimigo vencedor passar 
                reward -=  0.4
            # if reward < -1:
            #     reward = -1
        
        qValues = cast(torch.Tensor, playerModel(playerBoardState.unsqueeze(0)))
        target = qValues.clone().detach()
        
        if done:
            target[0, playerMove] = reward
        else:
            with torch.no_grad():
                nextPlayerState = board.clone() * player
                nextQ = cast(torch.Tensor, playerModel(nextPlayerState.unsqueeze(0)))[0]
                nextQ[[i for i in range(9) if i not in Utilities.valid_moves(nextPlayerState)]] = -1e9
                maxNextQ = nextQ.max()
                target[0, playerMove] = reward + gamma * maxNextQ

        loss = lossFn(qValues, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        debug_loss = (qValues[0, playerMove] - target[0, playerMove]).pow(2).item()

    epsilon = max(minEpsilon, epsilon * epsilonDecay)

    if(epoch % 500 == 0):
        enemyModel.load_state_dict(playerModel.state_dict())
    if(epoch % 1000 == 0):        
        print(
            f"epoch {epoch:>5d} | "
            f"loss {loss.item():8.5f} | "
            f"med_loss {total_loss:8.5f} | "
            f"epsilon {epsilon:.3f} | "
            f"target {target[0, playerMove]:+6.2f} | "
            f"Q {qValues[0, playerMove]:+6.2f} | "
            f"err {debug_loss:6.2f}"
        )
        total_loss = 0

torch.save(playerModel.state_dict(), "model_weights.pth")
evaluate(playerModel)