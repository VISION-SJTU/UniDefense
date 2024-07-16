import os
import yaml
import argparse

from engine import get_engine
from utils.misc import center_print

def arg_parser():
    parser = argparse.ArgumentParser(description="Training and Testing Script for UniDefense.")
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Specified the path of configuration file to be used.")
    parser.add_argument("--engine",
                        type=str,
                        default="UE",
                        choices=["FE", "OCIM", "UE"],
                        help="Specified the engine from 'FE' (Forgery), 'OCIM' (FAS), and 'UE' (UniAttack).")
    parser.add_argument("--local_rank", "-r",
                        type=int,
                        default=0,
                        help="Specified the node rank for distributed training or the gpu device for single "
                             "device testing.")
    parser.add_argument("--exp_id",
                        type=str,
                        default=None,
                        help="Overwrite exp id.")
    parser.add_argument("--ds_config",
                        type=str,
                        default=None,
                        help="Overwrite dataset config path.")
    parser.add_argument('--offline', action="store_true", help="Whether to use offline mode of wandb.")
    parser.add_argument('--test', action="store_true",
                        help="Whether to activate the test mode. Otherwise, it will be the training mode.")
    return parser.parse_args()


if __name__ == '__main__':
    arg = arg_parser()
    config = arg.config
    if arg.offline:
        os.environ["WANDB_MODE"] = "dryrun"
        center_print("NOTICE: Using OFFLINE mode for wandb.")

    with open(config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config["config"]["local_rank"] = arg.local_rank
    config["config"]["engine"] = arg.engine
    config["cfg_path"] = arg.config
    if arg.exp_id is not None:
        config["config"]["id"] = arg.exp_id
    if arg.ds_config is not None:
        config["data"]["file"] = arg.ds_config

    engine = get_engine(arg.engine)(config, stage="Test" if arg.test else "Train")
    if arg.test:
        engine.test()
    else:
        engine.train()
