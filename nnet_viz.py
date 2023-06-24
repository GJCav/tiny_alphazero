import sys
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv(
        "board", 9, 1, 
        offset="(0,0,0)", to="(0,0,0)", 
        height=64, depth=64, width=2, 
        caption="Board"
    ),
    to_Conv(
        "conv1", 9, 512, 
        offset="(5,0,0)", to="(board-east)", 
        height=64, depth=64, width=10,
        caption="Conv1"
    ),
    to_Conv(
        "conv2", 9, 512, 
        offset="(0,0,0)", to="(conv1-east)", 
        height=64, depth=64, width=10,
        caption="conv2"
    ),
    to_Conv(
        "conv3", 7, 512, 
        offset="(0,0,-1)", to="(conv2-east)", 
        height=48, depth=48, width=10,
        caption="conv3"
    ),
    to_Conv(
        "conv4", 5, 512, 
        offset="(0,0,-0.8)", to="(conv3-east)", 
        height=32, depth=32, width=10,
        caption="conv4"
    ),
    to_Conv(
        "tovec", 12800, 1, 
        offset="(5,0,0)", to="(conv4-east)", 
        height=2, depth=128, width=2,
        caption="vec"
    ),
    to_Conv(
        "fc1", 512, 1, 
        offset="(2,0,0)", to="(tovec-east)", 
        height=2, depth=64, width=2,
        caption="fc1"
    ),
    to_Conv(
        "fc2", 256, 1, 
        offset="(2,0,0)", to="(fc1-east)", 
        height=2, depth=32, width=2,
        caption="fc2"
    ),
    to_Conv(
        "fc3", 82, 1, 
        offset="(4,2,0)", to="(fc2-east)", 
        height=2, depth=20, width=2,
        caption="pi"
    ),
    to_Conv(
        "fc4", 1, 1, 
        offset="(4,-2,0)", to="(fc2-east)", 
        height=2, depth=2, width=2,
        caption="v"
    ),
    to_connection( "board", "conv1"),
    to_connection( "conv4", "tovec"),
    to_connection( "tovec", "fc1"),
    to_connection( "fc1", "fc2"),
    to_connection( "fc2", "fc3"),
    to_connection( "fc2", "fc4"),

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()