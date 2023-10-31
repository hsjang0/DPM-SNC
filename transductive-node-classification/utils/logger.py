import os

class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str, verbose=True):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')

        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()

        if verbose:
            print(str)


def set_log(config, is_train=True):

    data = config.data.data
    exp_name = config.method+'_'+config.model.model

    log_folder_name = os.path.join(*[data, exp_name])
    root = 'logs_train' 
    if not(os.path.isdir(f'./{root}/{log_folder_name}')):
        os.makedirs(os.path.join(f'./{root}/{log_folder_name}'))
    log_dir = os.path.join(f'./{root}/{log_folder_name}/')

    #print('-'*100)
    #print("Make Directory {} in Logs".format(log_folder_name))

    return log_folder_name, log_dir


def check_log(log_folder_name, log_name):
    return os.path.isfile(f'./logs_sample/{log_folder_name}/{log_name}.log')


def data_log(logger, config):
    logger.log(f'[{config.data.data}]   seed={config.seed}')


def model_log(logger, config):
    config_m = config.model
    model_log = f'nhid={config_m.nhid} layers={config_m.num_layers} '\
                f'linears={config_m.num_linears}'
    logger.log(model_log)


def start_log(logger, config):
    logger.log('-'*100)
    data_log(logger, config)
    logger.log('-'*100)


def train_log(logger, config):
    logger.log(f'lr={config.train.lr} weight_decay={config.train.weight_decay}')
    model_log(logger, config)
    logger.log('-'*100)

