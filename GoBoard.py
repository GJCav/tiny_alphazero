import numpy as np


class Stone:
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    ROB = 7

    @staticmethod
    def count_dict():
        return {Stone.BLACK: 0, Stone.WHITE: 0, Stone.EMPTY: 0, Stone.ROB: 0}


_neighbor_delta = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class Board:
    def __init__(self, n):
        assert n % 2 == 1
        self.n = n
        self.data = np.zeros((n, n))

        self.move_count = 0 # 落子数目
        self.rob = None # (i, j)  打劫，还得写在棋盘里面
        self.void_move = {Stone.BLACK: False, Stone.WHITE: False} # 虚着


    def copy(self, player=1):
        board = Board(self.n)
        board.data = self.data.copy()
        board.move_count = self.move_count
        board.rob = self.rob
        board.void_move = {
            Stone.BLACK: self.void_move[Stone.BLACK * player],
            Stone.WHITE: self.void_move[Stone.WHITE * player]
        }

        return board

    def __str__(self) -> str:
        ret = ''
        data = self.data
        for i in range(self.n):
            for j in range(self.n):
                if data[i, j] == Stone.EMPTY:
                    ret += '+'
                elif data[i, j] == Stone.BLACK:
                    ret += '○'
                elif data[i, j] == Stone.ROB:
                    ret += 'δ'
                else:
                    ret += '●'
            ret += '\n'
        return ret

    def load_from_numpy(self, a: np.ndarray):
        assert a.shape == (self.n, self.n)
        self.data = a

    def to_numpy(self):
        arr = self.data.copy()
        arr[np.where(arr == Stone.ROB)] = Stone.EMPTY # 忽略打劫
        return arr
        # Note: Copy if you don't want to mess up the original board.


    def add_stone(self, x, y, color):
        '''
        Add a stone to the board, and remove captured stones
        O(n^2) time complexity

        '''
        if x == None or y == None:
            # 虚着
            self.void_move[color] = True
            return
        
        if self.data[x, y] != Stone.EMPTY:
            assert False

        self.move_count += 1

        # 清除打劫
        if self.rob:
            self.data[self.rob] = Stone.EMPTY
            self.rob = None
        # 清除虚着
        self.void_move[Stone.BLACK] = False
        self.void_move[Stone.WHITE] = False

        self.data[x, y] = color
        pos = self.提子(x, y, -color)
        if np.count_nonzero(pos) == 1:
            # 出现打劫
            pos = np.where(pos == 1)
            pos = (pos[0][0], pos[1][0])
            self.rob = pos
            self.data[pos] = Stone.ROB

    def valid_moves(self, color):
        '''
        Return a list of avaliable moves
        O(n^4) time complexity ** TODO **

        @return: a list like [(0,0), (0,1), ...]
        '''
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        data = self.data
        actions = []

        for i in range(self.n):
            for j in range(self.n):
                if data[i, j] != Stone.EMPTY:
                    continue

                # 检查打劫
                if self.rob == (i, j):
                    continue

                # 检查禁着
                count, _ = self.__floodfill_empty(i, j)
                placeable = False
                if count[Stone.EMPTY] > 1:
                    placeable = True
                else:
                    data_bck = data.copy()
                    data[i, j] = color
                    self.提子(i, j, -color)
                    count, _ = self.__floodfill_stone(i, j, color)
                    data = data_bck
                    self.data = data_bck
                    if count[Stone.EMPTY] > 0:
                        placeable = True
                
                if not placeable:
                    continue

                actions.append((i, j))

        return actions
    

    def 提子(self, x, y, color):
        """
        -color 在 (x, y) 落子后，提掉可以提去的 color 的子。
        O(n^2) time complexity

        @return: 提掉的子的位置
        """

        # 这里不考虑打劫
        # 不想翻译哩 2333
        data = self.data

        # 提掉子的位置
        pos = np.zeros((self.n, self.n), dtype=bool)

        # 提子具有局域性，只会提掉和落子相邻的子。
        for dx, dy in _neighbor_delta:
            i = x + dx
            j = y + dy

            if i < 0 or i >= self.n or j < 0 or j >= self.n:
                continue

            if data[i, j] != color:
                continue
                
            count, vis = self.__floodfill_stone(i, j, color)
                
            if count[Stone.EMPTY] == 0:
                # 这个整体无气，整体提掉
                data[np.where(vis == 1)] = 0
                pos = np.logical_or(pos, vis)
                
        
        return pos
    
    def is_terminal(self):
        if self.move_count >= self.n * self.n * 4:
            return True
        if self.void_move[Stone.BLACK] and self.void_move[Stone.WHITE]:
            return True
        return False
    

    def get_scores(self):
        '''
        Compute score of players
        O(n^2) time complexity

        @return: a tuple (black_score, white_score)
        '''
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        score = Stone.count_dict()
        data = self.data

        def _d(s):
            # print(s)
            pass
        
        count = None
        visited = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                if data[i, j] not in [Stone.EMPTY, Stone.ROB]:
                    score[data[i, j]] += 1
                    continue

                if visited[i, j]: continue
                count, tmp = self.__floodfill_empty(i, j)
                count[Stone.EMPTY] += count[Stone.ROB]
                visited = np.logical_or(visited, tmp)
                if count[Stone.BLACK] == 0 and count[Stone.WHITE] != 0:
                    _d(f"while += {count[Stone.EMPTY]} at {i}, {j}")
                    score[Stone.WHITE] += count[Stone.EMPTY]
                elif count[Stone.BLACK] != 0 and count[Stone.WHITE] == 0:
                    _d(f"black += {count[Stone.EMPTY]} at {i}, {j}")
                    score[Stone.BLACK] += count[Stone.EMPTY]
        
        return [score[Stone.BLACK], score[Stone.WHITE]]


    def get_winner(self):
        score = self.get_scores()
        if score[0] > score[1]:
            return Stone.BLACK
        elif score[0] < score[1]:
            return Stone.WHITE
        else:
            return 1e-4
        
    
    def __floodfill_stone(self, ii, jj, color):
        return self.__floodfill(ii, jj, [color])
    
    def __floodfill_empty(self, ii, jj):
        return self.__floodfill(ii, jj, [Stone.EMPTY, Stone.ROB])
    
    def __floodfill(self, ii, jj, color=[]):
        visited = np.zeros((self.n, self.n), dtype=bool)
        count = Stone.count_dict()
        data = self.data

        def dfs(i, j):
            nonlocal count, visited

            count[int(data[i, j])] += 1
            if data[i, j] not in color: return

            visited[i, j] = True
            for dx, dy in _neighbor_delta:
                k = i + dx
                l = j + dy
                if k < 0 or k >= self.n or l < 0 or l >= self.n:
                    continue
                if visited[k, l]:
                    continue
                dfs(k, l)

        dfs(ii, jj)
        return count, visited


class BoardWizard:
    board: np.ndarray
    n: int
    
    fth: list[list[tuple[int, int]]]     # father of (i, j)
    count: list[list[dict[int, int]]]    # stone count dict at at (i, j)
    size: list[list[int]]                # size of union-set rooted  at size[i, j]


    def __init__(self, board: np.ndarray) -> None:
        shape = board.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]

        self.n = shape[0]
        self.board = board.copy()

        self.fth = [[(i, j) for j in range(self.n)] for i in range(self.n)]
        self.count = [[Stone.count_dict() for j in range(self.n)] for i in range(self.n)]
        self.size = [[1 for j in range(self.n)] for i in range(self.n)]


    def find(self, pos):
        h = pos
        while h != self.fth[h[0]][h[1]]:
            h = self.fth[h[0]][h[1]]

        while pos != h:
            pos, self.fth[pos[0]][pos[1]] = self.fth[pos[0]][pos[1]], h
        
        return h
    

    def connected(self, pos1, pos2):
        return self.find(pos1) == self.find(pos2)

    
    def connect(self, pos1, pos2):
        pos1 = self.find(pos1)
        pos2 = self.find(pos2)

        if pos1 == pos2:
            return
        
        if self.size[pos1[0]][pos1[1]] > self.size[pos2[0]][pos2[1]]:
            pos1, pos2 = pos2, pos1
        
        self.fth[pos1[0]][pos1[1]] = pos2
        self.size[pos2[0]][pos2[1]] += self.size[pos1[0]][pos1[1]]

    
    def dp(self):
        n = self.n
        board = self.board
        fth = self.fth
        count = self.count
        size = self.size

        for i in range(n):
            for j in range(n):
                if j >= 1:
                    pass