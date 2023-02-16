import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config


def main(work_type_args):
    args = Parser().parse()
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config = get_config(args.config, args.seed)

    if config.data.data == 'ppi-10' or config.data.data == 'ppi-20': # PPI-10, PPI-20
        from method_series.ddpm_trainer_huge_ppi import Trainer        
        trainer = Trainer(config) 
    elif config.data.data.startswith('ppi-') or config.data.data == 'dblp': # PPI-1, PPI-2, and DBLP
        from method_series.ddpm_trainer_huge import Trainer        
        trainer = Trainer(config) 
    else: # Pubmed, Cora, and Citeseer
        from method_series.ddpm_trainer import Trainer        
        trainer = Trainer(config) 
    trainer.train(ts)
            

if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    main(work_type_parser.parse_known_args()[0])
