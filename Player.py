import numpy as np

from MCTS import MCTS
from GoNNet import GoNNetWrapper


class RandomPlayer:
    def __init__(self, game, player):
        self.game = game
        self.player = player

    def __str__(self):
        return "Random Player"

    def play(self, board):
        valid_moves = self.game.get_valid_moves(board, self.player)
        a = np.random.choice(valid_moves.nonzero()[0])
        return a


class AlphaZeroPlayer:
    def __init__(self, game, checkpoint_path, num_sims, C, chk_abs_path = None, player=1) -> None:
        self.nnet = GoNNetWrapper(game)
        if chk_abs_path is None:
            self.checkpoint_path = checkpoint_path
            self.nnet.load_checkpoint(self.checkpoint_path)
        else:
            self.checkpoint_path = chk_abs_path
            self.nnet.load_checkpoint(path=chk_abs_path)
        self.mcts =  MCTS(game, self.nnet, num_sims, C)
        self.player = player
    
    def __str__(self):
        return f"AlphaZero Player({self.checkpoint_path})"

    def play(self, board):
        return np.argmax(self.mcts.get_action_prob(board, self.player))

class FastEvalPlayer:
    def __init__(self, mcts):
        self.mcts = mcts
        self.player = 0
    def play(self, board):
        return np.argmax(self.mcts.get_action_prob(board, self.player))