import torch
import random
from typing import cast
from Model import TicTacToeModel
from evaluate import evaluate

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # Check if a GPU is available, otherwise use CPU
print(f"Running on {device}")

playerModel = TicTacToeModel().to(device)
#playerModel.load_state_dict(torch.load('model_weights.pth')) # Remove comment to continue training from saved weights
playerModel.train()
# evaluate(playerModel, device=device) 

enemyModel = TicTacToeModel().to(device) # Make a copy of the player model to act as the enemy
enemyModel.load_state_dict(playerModel.state_dict())
enemyModel.eval()

# == HYPERPARAMETERS ==
gamma = 0.95 # This is the discount factor used to calculate future rewards. A value of 0.95 means that future rewards are worth slightly less than immediate rewards.
epochs = 16000 # Number of games to play
epsilon = 1.0 # The probability of choosing a random action instead of the best one. This helps to explore new strategies
minEpsilon = 0 # The minimum value of epsilon. This helps to ensure that the model eventually explores all possible strategies
epsilonDecay = 0.9998 # The rate at which epsilon decreases. This helps to balance exploration and exploitation
matchPenaltyDecay = -0.01 # The rate at which the match penalty decreases. This helps to balance exploration and exploitation
learningRate = 1e-3 # The learning rate used to update the model's weights
lossFn = torch.nn.MSELoss() # The loss function used to calculate the difference between the predicted and actual values
optimizer = torch.optim.Adam(playerModel.parameters(), lr=learningRate) # The optimizer used to update the model's weights

class Utilities():
    wins = [
        [0,1,2], [3,4,5], [6,7,8],  # lines
        [0,3,6], [1,4,7], [2,5,8],  # columns
        [0,4,8], [2,4,6]            # diagonals
    ]

    def checkWin(board: torch.Tensor): 
        """
        Checks if there is a winner on the board.
        Returns:
           int: Returns the winning player (1 or -1), a draw (2) or no winner yet (0).
        """
        for a,b,c in Utilities.wins:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a].item() # Returns the winning player (1 or -1)
        if torch.all(board != 0):
            return 2 # Returns a draw (2)
        return 0 # Returns no winner yet (0)

    def valid_moves(board: torch.Tensor):
        """
        Checks for valid moves on the board.
        Returns:
            List[int]: A list of valid moves (empty spaces) on the board.
        """
        return [i for i in range(9) if board[i] == 0] 
    
    def hasWinningMove(board: torch.Tensor, player: int):
        """
        Checks if the player has a winning move
        Returns:
            bool: Returns True if the player has a winning move, otherwise returns False.
        """
        for move in Utilities.valid_moves(board):
            tempBoard = board.clone()
            tempBoard[move] = player
            if(Utilities.checkWin(tempBoard) == player): return True
        return False
    
    def isPlayerWin(player: int, matchResult: int):
        """
        Checks if the match result is a win
        Returns:
            bool: Returns True if the match result is a win, otherwise returns False.
        """
        return player == matchResult
    
    def isDraw(matchResult: int):
        """
        Checks if the match result is a draw
        Returns:
            bool: Returns True if the
        """
        return matchResult == 2
    
total_loss = 0 # total loss of the model during training
toggle_first = False # toggle to switch who will start first in the match

print(
    f"gamma {gamma} | " 
    f"epochs {epochs} | " 
    f"epsilon {epsilon} | "
    f"minEpsilon {minEpsilon} | "
    f"epsilonDecay {epsilonDecay} | "
    f"learningRate {learningRate}"
)

# == TRAINING LOOP ==
for epoch in range(epochs):
    player = 1
    enemy = -player
    board = torch.zeros(9, device=device)
    matchPenalty = 0

    # Choose who will start first in the match
    if(toggle_first):
        firstMove = random.choice(Utilities.valid_moves(board))
        board[firstMove] = enemy
        toggle_first = False
    else:
        toggle_first = True

    done = False # Flag to indicate when the game is over

    # == MATCH LOOP ==
    while not done:
        validMoves = Utilities.valid_moves(board)

        # == PLAYER MOVE ==
        if(len(validMoves) > 0):
            playerBoardState = board.clone() * player # Multiply by player to get player's perspective
            if(random.random() < epsilon): # Epsilon: Decide whether to play randomly or based on the model's prediction. 
                playerMove = random.choice(validMoves)
            else:
                with torch.no_grad(): # Desactive gradient computation to avoid updating the model's weights during inference. 
                    playerMovePossibilities = cast(torch.Tensor, playerModel(playerBoardState.unsqueeze(0))[0]) # Generate model predictions for all possible moves. 
                    playerMovePossibilities[[i for i in range(9) if i not in validMoves]] = -1e9 # Set invalid moves to a very low value (-1e9) to avoid selecting them. 
                    playerMove = torch.argmax(playerMovePossibilities).item() # Select the move with the highest predicted value. 
            board[playerMove] = player # Make the player's move on the board.

        # == REWARD CALCULATION ==
        reward = 0
        matchResult = Utilities.checkWin(board)
        if(Utilities.isPlayerWin(player, matchResult)):
            reward = 1 # Win for the player.
            done = True        
        elif(Utilities.isDraw(matchResult)):
            reward = 0 # Draw for the player.
            done = True

        # == ENEMY MOVE ==
        validMoves = Utilities.valid_moves(board)
        if(len(validMoves) > 0 and not done):            
            enemyBoardState = board.clone() * enemy # Multiply by enemy to get enemy's perspective. 
            if(random.random() < .1):
                enemyMove = random.choice(validMoves)
            else:
                with torch.no_grad():
                    enemyMovePossibilities = cast(torch.Tensor, enemyModel(enemyBoardState.unsqueeze(0)))[0]
                    enemyMovePossibilities[[i for i in range(9) if i not in validMoves]] = -1e9
                    enemyMove = torch.argmax(enemyMovePossibilities).item()
            board[enemyMove] = enemy
        
            # == REWARD CALCULATION FOR ENEMY ==
            matchResult = Utilities.checkWin(board)
            if(Utilities.isPlayerWin(enemy, matchResult)):
                reward = -1 # Loss for the player.
                done = True
            elif(Utilities.isDraw(matchResult)):
                reward = 0.5 # Enemy draw for the player.
                done = True
        
        # == MATCH PENALTY ==
        if(not done):
            reward = matchPenalty
            if(len(Utilities.valid_moves(board)) < 7):
                matchPenalty += matchPenaltyDecay # Penalize the player for having less moves left.
            if(Utilities.hasWinningMove(playerBoardState, player)):
                reward -= 0.5 # Penalize the player for losing a winning move.
            if(Utilities.hasWinningMove(board, enemy)): 
                reward -= 0.4 # Penalize the player for allowing the enemy to have a winning move.
        
        
        qValues = cast(torch.Tensor, playerModel(playerBoardState.unsqueeze(0))) # Get the Q-values for the current state. (This board is before both players have made their moves)
        target = qValues.clone().detach() # Clone the Q-values and detach them from the computation graph (detach() is used to prevent the gradient from flowing back through this tensor).
        
        if done:
            target[0, playerMove] = reward # Set the target Q-value for the action taken to
        else:
            with torch.no_grad():
                nextPlayerState = board.clone() * player # Clone the board and multiply it by the main player (This board is after both players have made their moves)
                nextQ = cast(torch.Tensor, playerModel(nextPlayerState.unsqueeze(0)))[0] # Get the Q-values for the next state.
                nextQ[[i for i in range(9) if i not in Utilities.valid_moves(nextPlayerState)]] = -1e9 
                maxNextQ = nextQ.max() # Get the maximum Q-value for the next state.
                target[0, playerMove] = reward + gamma * maxNextQ # Q(state, action) expected total reward if the player make the action (a) into state (s)

        loss = lossFn(qValues, target) # Calculate the loss between the Q-values and the target
        optimizer.zero_grad() # Clear the gradients before running the backward pass.        
        loss.backward() # Run the backward pass to compute gradients.
        optimizer.step() # Update the weights of the model.
        total_loss += loss.item() # Accumulate the loss for each epoch.
        debug_loss = (qValues[0, playerMove] - target[0, playerMove]).pow(2).item() # Calculate the debug loss for each epoch.

    epsilon = max(minEpsilon, epsilon * epsilonDecay) # Decay the epsilon value.

    if(epoch % 500 == 0):
        enemyModel.load_state_dict(playerModel.state_dict()) # Copy the weights of the main model to the enemy model. Self-play helps to stabilize training.

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

torch.save(playerModel.state_dict(), "model_weights.pth") # Save the model weights.
evaluate(playerModel)