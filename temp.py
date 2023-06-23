from GoNNet import *
from GoGame import *

g = GoGame(5)
net = GoNNetWrapper(g)

b = Board(5)
b.load_from_numpy(np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 0,-1, 1, 0],
    [ 0,-1, 1, 0, 0],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1],
]))

print(net.predict(b.data))