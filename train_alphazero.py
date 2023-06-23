import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import datetime
import time
import json

import numpy as np
from tqdm import tqdm

from pit import test_multi_match
from Player import FastEvalPlayer, RandomPlayer
from MCTS import MCTS

log = logging.getLogger(__name__)


class Trainer():
    """
    """

    def __init__(self, game, nnet, config):
        self.game = game
        self.next_net = nnet
        self.last_net = self.next_net.__class__(self.game)
        self.config = config
        self.mcts = MCTS(self.game, self.next_net, self.config.num_sims, self.config.cpuct)
        self.train_data_packs = []

    def collect_single_game(self):
        """
        Collect self-play data for one game.
        
        @return game_history: A list of (board, pi, z)
        """
        # create a New MCTS 
        self.mcts = MCTS(self.game, self.next_net, self.config.num_sims, self.config.cpuct)
        self.mcts.train()

        game_history = []
        board = self.game.reset()
        current_player = 1
        current_step = 0

        # self-play until the game is ended
        while True:
            current_step += 1
            pi = self.mcts.get_action_prob(board, current_player)
            datas = self.game.get_transform_data(self.game.get_board(board, current_player), pi)
            for b, p in datas:
                game_history.append([b.to_numpy(), current_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board = self.game.next_state(board, current_player, action)
            current_player *= -1
            game_result = self.game.is_terminal(board, current_player)

            if game_result != 0:  # Game Ended
                return [(x[0], x[2], game_result * ((-1) ** (x[1] != current_player))) for x in game_history]

    def train(self):
        """
        Main Training Loop of AlphaZero
        each iteration:
            * Collect data by self play
            * Train the network
            * Pit the new model against the old model
                If the new model wins, save the new model, and evaluate the new model
                Otherwise, reject the new model and keep the old model
            
        """
        log_name = "itr_info_" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + ".mutijson"


        for i in range(1, self.config.max_training_iter + 1):

            log.info(f'Starting Iter #{i} ...')
            itr_info = {
                "itr": i,
                "time": time.time(),
                "loss": 1000,
                "pit_random": None,
                "pit_last": None,
                "accept": False
            }

            data_pack = deque([])
            T = tqdm(range(self.config.selfplay_each_iter), desc="Self Play")
            for _ in T:
                game_data = self.collect_single_game()
                data_pack += game_data
                r = game_data[0][-1]
                T.set_description_str(f"Self Play win={r}, len={len(game_data)}")

            self.train_data_packs.append(data_pack)


            trainExamples = []
            for e in self.train_data_packs:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.next_net.save_checkpoint(folder=self.config.checkpoint_folder, filename='temp.pth.tar')
            self.last_net.load_checkpoint(folder=self.config.checkpoint_folder, filename='temp.pth.tar')

            loss_rec = self.next_net.train(trainExamples)
            loss = loss_rec[-1]
            itr_info["loss"] = loss

            next_mcts = MCTS(self.game, self.next_net, self.config.num_sims, self.config.cpuct)
            last_mcts = MCTS(self.game, self.last_net, self.config.num_sims, self.config.cpuct)

            ######################################
            #        YOUR CODE GOES HERE         #
            ###################################### 
            # Pitting against last version, and decide whether to save the new model

            next_player = FastEvalPlayer(next_mcts)
            last_player = FastEvalPlayer(last_mcts)
            n_test = self.config.get("n_test", 100)

            accept_new = False
            __against_rdm = 1.05
            last_rate = getattr(self, "last_rate", None)

            # first pit against random player
            if last_rate is not None and last_rate * __against_rdm >=1:
                log.info(f"last_rate * {__against_rdm} >= 1, skip and decrease last_rate")
                self.last_rate = last_rate / __against_rdm
            else:
                log.info('Pitting against random player...')
                next_player = FastEvalPlayer(next_mcts)
                rand_player = RandomPlayer(self.game, 1)
                next_win, rdm_win, draw = test_multi_match(
                    player1=next_player, player2=rand_player, game=self.game, n_test=n_test
                )

                itr_info["pit_random"] = (next_win, rdm_win, draw)

                win_rate = next_win / (next_win + rdm_win + draw)
                if last_rate is None:
                    log.info(f"first itr, skip")
                elif last_rate * 1.05 < win_rate: 
                    log.info(f"against rdm player: {last_rate} -> {win_rate}")
                    accept_new = True

                self.last_rate = max(win_rate, last_rate if last_rate is not None else 0.0)

            # if necessary pit against last version
            if not accept_new:
                log.info('Pitting against last version...')
                
                next_win, last_win, draw = test_multi_match(
                    player1=next_player, player2=last_player, game=self.game, n_test=n_test
                )

                itr_info["pit_last"] = (next_win, last_win, draw)

                update_threshold = self.config["update_threshold"]
                checkpoint_folder = self.config["checkpoint_folder"]
                win_rate = next_win / (next_win + last_win + draw)
                if win_rate > update_threshold:
                    log.info(f"against last version: {win_rate} > {update_threshold}")
                    accept_new = True

            itr_info["accept"] = accept_new
            if accept_new:
                log.info("Accept new model.")
                self.next_net.save_checkpoint(
                    folder=checkpoint_folder, filename=f"checkpoint_{i}.pth.tar"
                )
                self.next_net.save_checkpoint(
                    folder=checkpoint_folder, filename="best.pth.tar"
                )
            else:
                log.info("Reject new model.")
                self.next_net.load_checkpoint(
                    folder=checkpoint_folder, filename="temp.pth.tar"
                )

            with open(os.path.join(checkpoint_folder, log_name), "a") as f:
                f.write(json.dumps(itr_info) + "\n")

            
            
