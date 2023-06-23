import math
import pickle
import os

import numpy as np
from util import *


class MCTS:
    def __init__(self, game, nnet, num_sims, C):
        self.game = game
        self.num_sims = num_sims
        self.nnet = nnet
        self.C = C
        self.training = True
        self.W_state_action = {}  # stores total action-value
        # self.Q_state_action = {}  # stores mean action-value, no need to store it
        # stores times edge (state,action) was visited
        self.N_state_action = {}
        # stores times board state was visited
        self.N_state = {}
        # stores policy
        self.P_state = {}

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_action_prob(self, board, player):
        board = self.game.get_board(board, player)
        if self.training:
            for i in range(self.num_sims):
                self.search(board)

        s = self.game.get_string(board)
        counts = np.array([
            self.N_state_action[(s, a)] if (s, a) in self.N_state_action 
            else 0 
            for a in range(self.game.action_size())
        ])
        sum_count = counts.sum()
        if sum_count:
            probs = counts / sum_count
        else:
            probs = np.ones(len(counts), dtype=float)/len(counts)
        return probs
    

    def UCB_sample(self, board):
        raise NotImplementedError

        weights = []
        moves = []
        move_mask = self.game.get_valid_moves(board, 1)
        s = self.game.get_string(board)
        for act in range(self.game.action_size()):
            if not move_mask[act]:
                continue

            U = self.W_state_action[(s, act)]
            N = self.N_state_action[(s, act)]
            P = self.N_state[s]
            w = U / N + self.C * math.sqrt(math.log(P+1) / N)

            moves.append(act)
            weights.append(w)
        
        weights = np.array(weights)
        # print(weights)
        move = np.random.choice(moves, p=weights/np.sum(weights))
        return move
    

    def PUCT_sample(self, board):
        weights = []
        moves = []
        move_mask = self.game.get_valid_moves(board, 1)

        P_arr, value = self.nnet.predict(board.data)
        value = value[0]
        P_arr *= move_mask
        P_arr /= np.sum(P_arr)
        
        s = self.game.get_string(board)
        for act in range(self.game.action_size()):
            if not move_mask[act]:
                continue

            N = self.N_state_action[(s, act)]
            W = self.W_state_action[(s, act)]
            Q = W / N
            P = P_arr[act]

            U = self.C * P * math.sqrt(self.N_state[s]) / (1 + N)
            w = Q + U

            moves.append(act)
            weights.append(w)
        
        weights = np.array(weights)
        move = moves[np.argmax(weights)]
        return move, value


    def search(self, board, **kwargs):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound.
        """
        # use the current board state to get a unique string representation as a key
        s = self.game.get_string(board)

        # TODO handles leaf node
        ##############################
        # YOUR CODE GOES HERE
        ##############################
        g = self.game.is_terminal(board, 1)
        if g != 0:
            return g
            

        # TODO pick an action with the highest upper confidence bound (UCB)
        ##############################
        # YOUR CODE GOES HERE
        next_board = None # compute the next board after executing the best action here
        ##############################

        # TODO update Q_state_action, N_state_action, and N_state
        ##############################
        # YOUR CODE GOES HERE
        ##############################
        
        valid_move_mask = self.game.get_valid_moves(board, 1)
        unexpanded_move_mask = valid_move_mask.copy()
        for act in range(self.game.action_size()):
            if not valid_move_mask[act]:
                continue
            if (s, act) in self.N_state_action:
                unexpanded_move_mask[act] = 0
        
        if not np.any(unexpanded_move_mask):
            # all moves have been expanded, select a move
            move, value = self.PUCT_sample(board)
            next_board = self.game.next_state(board, 1, move)
            next_board = self.game.get_board(next_board, -1)

            if kwargs.get("__depth", 0) > 800:
                breakpoint()

            winner = -self.search(next_board, __depth=kwargs.get('__depth', 0)+1)
        else:
            # select a move that has not been expanded
            # sample based on the nnet

            P, _ = self.nnet.predict(board.data)
            P *= unexpanded_move_mask
            P /= np.sum(P)

            move = np.random.choice(np.arange(self.game.action_size()), p=P)
            next_board = self.game.next_state(board, 1, move)
            next_board = self.game.get_board(next_board, -1)
            ns = self.game.get_string(next_board)

            self.N_state_action[(s, move)] = 0
            self.W_state_action[(s, move)] = 0

            _, value = self.nnet.predict(next_board.data)
            value = value[0]
            winner = -value
            self.N_state[ns] = 1
        
        self.N_state[s] = self.N_state.get(s, 0) + 1
        self.N_state_action[(s, move)] = self.N_state_action.get((s, move), 0) + 1
        self.W_state_action[(s, move)] = self.W_state_action.get((s, move), 0) + winner

        return winner

    def save_params(self, file_name="mcts_param.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_params(self, file_name="mcts_param.pkl"):
        if not os.path.exists(file_name):
            print(f"Parameter file {file_name} does not exist, load failed!")
            return False
        with open(file_name, "rb") as f:
            self.__dict__ = pickle.load(f)
            print(f"Loaded parameters from {file_name}")
        return True
