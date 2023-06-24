import logging
import coloredlogs

from train_alphazero import Trainer
from GoGame import GoGame as Game
from Player import *
from GoNNet import GoNNetWrapper as nn
from util import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

BOARD_SIZE = 7
args = dotdict({
    'max_training_iter': 5000,                     # 训练主循环最大迭代次数
    'selfplay_each_iter': 30,                     # 每次训练迭代自我对弈次数
    'num_sims': 30,                                # MCTS搜索的模拟次数
    'n_test': 30,                                  # 评估时的对弈次数

    'update_threshold': 0.51,                      # 更新模型胜率阈值
    'cpuct': 1,                                    # MCTS探索系数
    'cuda': True,                                  # 启用CUDA
    'checkpoint_folder': './7x7_rst/',                # 模型保存路径
    'load_model': False,                           # 是否加载模型
    'load_folder_file': ('temp','best.pth.tar'),   # 加载模型的路径
    'pit_with': RandomPlayer(Game(BOARD_SIZE), 1), # 评估时的对手
})


# args = dotdict({
#     'max_training_iter': 5000,                     # 训练主循环最大迭代次数
#     'selfplay_each_iter': 100,                     # 每次训练迭代自我对弈次数
#     'update_threshold': 0.51,                      # 更新模型胜率阈值
#     'update_match_cnt': 30,                        # 计算更新模型胜率阈值的对弈次数
#     'eval_match_cnt': 10,                          # 每次更新模型后，进行评估的对弈次数
#     'num_sims': 50,                                # MCTS搜索的模拟次数
#     'cpuct': 1,                                    # MCTS探索系数
#     'cuda': True,                                  # 启用CUDA
#     'checkpoint_folder': './temp/',                # 模型保存路径
#     'load_model': False,                           # 是否加载模型
#     'load_folder_file': ('temp','best.pth.tar'),   # 加载模型的路径
#     'pit_with': RandomPlayer(Game(BOARD_SIZE), 1), # 评估时的对手
# })


def main():
    set_seed_everywhere(1)
    log.info('Loading %s...', Game.__name__)
    g = Game(BOARD_SIZE)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Build Trainer...')
    c = Trainer(g, nnet, args)

    log.info('Start Training ...')
    c.train()


if __name__ == "__main__":
    main()
