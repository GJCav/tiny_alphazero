import numpy as np
from GoBoard import Board, Stone


class GoGame:
    def __init__(self, n=3):
        assert n % 2 == 1
        self.n = n

    def reset(self):
        """
        Reset the game.
        """
        return Board(self.n)
    
    def obs_size(self):
        """
        Size of the board.
        """
        return self.n, self.n

    def action_size(self):
        """
        Number of all possible actions.
        """
        return self.n * self.n + 1 # the extra 1 is for 'pass'(虚着), a action for doing nothing

    def get_board(self, board: Board, player: int) -> Board:
        """
        Convert given board to a board from player1's perspective.
        If current player is player1, do nothing, else, reverse the board.
        This can help you write neater code for search algorithms as you can go for the maximum return every step.

        @param board: the board to convert
        @param player: 1 or -1, the player of current board

        @return: a board from player1's perspective
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################


        # easy to make bug
        #   get next board should inherit void_move and other information
        #   from current board
        new_board = board.copy(player)
        return new_board

    def next_state(self, board: Board, player: int, action: int):
        """
        Get the next state by executing the action for player.
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        # Note: don't forget 'pass'
        
        if action == self.n * self.n:
            # 虚着
            x, y = None, None
        else:
            x, y = action//self.n, action%self.n

        next_board = board.copy()
        next_board.add_stone(x, y, player)
        return next_board

    def get_valid_moves(self, board: Board, player: int):
        """
        Get a binary vector of length self.action_size(), 1 for all valid moves.
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        valid_moves = np.zeros(self.action_size())
        pos = board.valid_moves(player)
        for x, y in pos:
            valid_moves[x*self.n+y] = 1
        valid_moves[-1] = 1 # 这是虚着
        return valid_moves
    

    def get_transform_data(self, board: Board, policy):
        """
        Rotate and flip the board and corresponding policy vector. 
        This method adds more examples to the training dataset, 
        accelerating the training process.
        
        board: current board
        policy: policy vector
        
        return: [(board1, policy1), (board2, policy2), ...]
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        
        # you can simply return [(board, policy)] 
        # if you don't want to use this method
        return [(board, policy)]


    @staticmethod
    def is_terminal(board: Board, player: int):
        """
        Check whether the game is over.
        @return: 1 or -1 if player or opponent wins, 1e-4 if draw, 0 if not over
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        # Note: end game when two consecutive passes or too many moves, compare scores of two players.
        
        if board.is_terminal():
            return board.get_winner()
        else:
            return 0

    @staticmethod
    def get_string(board: Board):
        """
        Convert the board to a string.
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        # Note: different board(Game Statue) must return different string
        
        return str(board)

    @staticmethod
    def display(board: Board):
        """
        Print the board to console.
        """
        print(str(board))
