from flask import Flask, render_template, request, jsonify
from Model import TicTacToeModel
from TicTacToe import TicTacToe

app = Flask(__name__)

def start(model: TicTacToeModel, device="cpu"):
    ticTacToe = TicTacToe(model, device);
    game_state = {
        "toggle_first": True,
        "player": -1,
        "ai": 1
    }

    @app.route("/")
    def index():
        return render_template("index.html")
    
    @app.route("/move", methods=["POST"])
    def move():
        data = request.json
        idx = data["index"]

        if(ticTacToe.board[idx] != 0):
            return jsonify({"error": "invalid"})
        
        ticTacToe.move(game_state['player'], idx)
        winner = ticTacToe.checkWin() 
        if winner != 0:
            return jsonify({
                "board": ticTacToe.board.tolist(), 
                "winner": winner
            })
        
        aiMove = ticTacToe.chooseAction(0)
        ticTacToe.move(game_state['ai'], aiMove)

        winner = ticTacToe.checkWin()
        return jsonify({
            "board": ticTacToe.board.tolist(), 
            "winner": winner
        })

    @app.route("/reset", methods=["POST"])
    def reset():
        ticTacToe.reset()
        if game_state['toggle_first']:
            ticTacToe.move(game_state['ai'], ticTacToe.chooseAction(0))
            game_state['toggle_first'] = False
        else:
            game_state['toggle_first'] = True

        return jsonify({
            "board": ticTacToe.board.tolist(),
            "winner": ticTacToe.checkWin()
        })

    print("Running on port 3000")
    app.run(host="0.0.0.0", port=3000, debug=True)

