import torch
import random
from TicTacToe import TicTacToe

def evaluate(model, games=1000, device='cpu'):
    env = TicTacToe(agent=model, device=device)
    wins = draws = losses = 0

    for _ in range(games):
        player = 1
        env.reset()
        env.move(
            player * (-1 if random.random() > .5 else 1),
            random.choice(env.valid_moves())
        )
        done = False

        while not done:
            state = env.board.clone() * player

            with torch.no_grad():
                q = model(state.unsqueeze(0))[0]
                q[[i for i in range(9) if i not in env.valid_moves()]] = -1e9
                action = q.argmax().item()

            env.move(player, action)
            r = env.checkWin()
            if r != 0:
                if r == player:
                    wins += 1
                elif r == 2:
                    draws += 1
                else:
                    losses += 1
                break

            # oponente aleatório
            if(random.random() > .1):
                env.move(-player, random.choice(env.valid_moves()))
            else:
                with torch.no_grad():
                    q = model((env.board.clone() * -player).unsqueeze(0))[0]
                    q[[i for i in range(9) if i not in env.valid_moves()]] = -1e9
                    action = q.argmax().item()
                    env.move(-player, action)
            r = env.checkWin()
            if r != 0:
                if r == player:
                    wins += 1
                elif r == 2:
                    draws += 1
                else:
                    losses += 1
                break
                
    print(f"Wins: {wins} {wins/games * 100:.2f}% | Draws: {draws} {draws/games * 100:.2f}% | Losses: {losses} {losses/games * 100:.2f}%")