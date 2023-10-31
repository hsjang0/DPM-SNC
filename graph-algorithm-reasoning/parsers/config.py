import yaml
from easydict import EasyDict as edict


def get_config(config):
    config_dir = f'./config/{config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = 42
    return config