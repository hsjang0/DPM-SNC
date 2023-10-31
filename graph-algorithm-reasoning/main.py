import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from method_series.dpm_trainer import Trainer


def main(work_type_args):
    args = Parser().parse()
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config = get_config(args.config)
    trainer = Trainer(config) 
    ckpt = trainer.train(ts)
            
if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    main(work_type_parser.parse_known_args()[0])
