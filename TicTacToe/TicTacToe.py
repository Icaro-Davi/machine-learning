import random
import torch
from typing import List
from Model import TicTacToeModel

class TicTacToe():
    def __init__(self, agent=TicTacToeModel, device="cpu", board=torch.zeros(9)):
        self.agent = agent.to(device)
        self.board = board.to(device)
        self.wins = [
            [0,1,2], [3,4,5], [6,7,8],  # linhas
            [0,3,6], [1,4,7], [2,5,8],  # colunas
            [0,4,8], [2,4,6]            # diagonais
        ]

    def reset(self) -> torch.Tensor:
        self.board.fill_(0.)
        return self.board

    def valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def checkWin(self): 
        """
        Verifica o estado de um jogo da velha.

        Args:
            board (List[int]): O tabuleiro linearizado (9 posições), com valores 0 (vazio), 1 ou -1.
        
        Returns:
            int: 1 ou -1 para vitória, 2 para empate, 0 para continuar.
        """
        for a,b,c in self.wins:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a].item() # 1 ou -1
        
        if torch.all(self.board != 0):
            return 2 # empate
        
        return 0 # continua
    
    def chooseAction(self, epsilon: float, train=False) -> int:
        """
        Escolhe uma jogada na tabela
        Args:
            agent (TicTacToeModel): Modelo utilizado para a ação
            state (Tensor): O tabuleiro linearizado (9 posições), com valores 0 (vazio), 1 ou -1.
            epsilon (float): Valor que representa uma jogada aleatoria 
        Returns:
            int: Posição no tabuleiro (board) que foi jogada
        """
        validMoves = [i for i in range(9) if self.board[i] == 0] # Busca as posições válidas restantes

        if train and random.random() < epsilon:
            return random.choice(validMoves) # escolhe uma jogada aleatoria sem depender do agent (modelo)
        
        with torch.set_grad_enabled(train):
            clonedBoard = self.board.clone()
            qValues = self.agent(clonedBoard.unsqueeze(0))[0] # Envolve o tensor (state) em uma nova camada de colchetes [a,b,c...] -> [[a,b,c...]]

            for i in range(9): # percorre todas as casas do tabuleiro
                if i not in validMoves: # se não achar o index subsitui o qValues[x] para um valor que represente inválido (-9999)
                    qValues[i] = -1e9

            action = torch.argmax(qValues).item() # retorna o maior valor (probabilidade) de dentro de um tensor
        return action
    
    def move(self, player=int, index=int):
        self.board[index] = player
    

