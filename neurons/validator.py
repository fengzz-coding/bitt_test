# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import argparse
import asyncio
import time

import torch
import typing
import bittensor as bt

from flockoff.utils.git import check_and_update_code
import os
import argparse
import asyncio
import torch
import typing

import numpy as np


from flockoff import constants
from flockoff.utils.chain import assert_registered, read_chain_commitment

from flockoff.validator.database import ScoreDB



class Validator:
    @staticmethod
    def config():
        bt.logging.info("Parsing command line arguments")
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--blocks_per_epoch",
            type=int,
            default=360,
            help="Number of blocks to wait before setting weights.",
        )
        parser.add_argument(
            "--miner_sample_size",
            type=int,
            default=10,
            help="Number of miners to sample for each block.",
        )
        parser.add_argument("--netuid", type=int, required=True, help="The subnet UID.")

        parser.add_argument(
            "--cache_dir",
            type=str,
            default="~/data/hf_cache",
            help="Directory to store downloaded model files.",
        )

        parser.add_argument(
            "--data_dir",
            type=str,
            default="~/data/training_data",
            help="Directory to store miner datasets.",
        )

        parser.add_argument(
            "--eval_data_dir",
            type=str,
            default="~/data/eval_data",
            help="Directory to store evaluation datasets.",
        )

        parser.add_argument(
            "--block_threshold",
            type=int,
            default=50,
            help="Number of blocks before epoch end to set weights.",
        )

        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        bt.logging.debug(f"Parsed config: {config}")
        return config

    def __init__(self):
        bt.logging.info("Initializing validator")
        self.config = Validator.config()

        bt.logging.info("Checking git branch")

        self.cnt = 0

        if self.config.cache_dir and self.config.cache_dir.startswith("~"):
            self.config.cache_dir = os.path.expanduser(self.config.cache_dir)

        if self.config.data_dir and self.config.data_dir.startswith("~"):
            self.config.data_dir = os.path.expanduser(self.config.data_dir)

        if self.config.eval_data_dir and self.config.eval_data_dir.startswith("~"):
            self.config.eval_data_dir = os.path.expanduser(self.config.eval_data_dir)

        bt.logging(config=self.config)
        bt.logging.info(f"Starting validator with config: {self.config}")

        # === Bittensor objects ====
        bt.logging.info("Initializing wallet")
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet initialized: {self.wallet}")
        bt.logging.info("Initializing subtensor")
        try:
            self.subtensor = bt.subtensor(config=self.config)
            bt.logging.info(f"Subtensor initialized: {self.subtensor}")
            bt.logging.info(f"Connected to network: {self.subtensor.network}")
            bt.logging.info(f"Chain endpoint: {self.subtensor.chain_endpoint}")
        except Exception as e:
            bt.logging.error(f"Failed to initialize subtensor: {e}")
            raise

        self.dendrite = bt.dendrite(wallet=self.wallet)

        bt.logging.info(f"Fetching metagraph for netuid: {self.config.netuid}")
        self.metagraph: bt.metagraph = self.subtensor.metagraph(self.config.netuid)

        bt.logging.info("Checking if wallet is registered on subnet")
        self.uid = 0

        bt.logging.info("Initializing weights tensor")
        self.weights = torch.zeros_like(torch.tensor(self.metagraph.S))
        bt.logging.info(f"Weights initialized with shape: {self.weights.shape}")

        self.uids_to_eval: typing.Dict[str, typing.List] = {}
        bt.logging.info("Initializing score database")
        self.score_db = ScoreDB("scores.db")
        bt.logging.info("Score database initialized")
        self.rng = np.random.default_rng()
        bt.logging.info("Validator initialization complete")

        self.last_competition_hash = None
        tempo = self.subtensor.tempo(self.config.netuid)
        self.last_submitted_epoch = (
                self.subtensor.get_next_epoch_start_block(self.config.netuid) - tempo
        )

        bt.logging.info("Validator ready to run")

    async def run_step(self):
        bt.logging.info("Starting run step")
        self.cnt += 1
        bt.logging.info(f"Now the cnt is {self.cnt}")
        check_and_update_code()
        self.score_db.set_revision("23", "test")
        time.sleep(10)

    async def run(self):
        while True:
            await self.run_step()


if __name__ == "__main__":
    asyncio.run(Validator().run())