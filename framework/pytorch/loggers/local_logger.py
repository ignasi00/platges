
import numpy as np
#import pandas as pd


LOSS = 'loss'
NTOKENS = 'ntokens'
NITERATIONS = 'niterations'


class LocalLogger():

    def __init__(self, metric_funct_dict, size_dataset=None, prefix=None):
        self.metric_funct_dict = metric_funct_dict
        self.epoch_log = {name : 0 for name in self.metric_funct_dict.keys()}
        self.epoch_log[LOSS] = 0
        self.epoch_log[NTOKENS] = 0
        self.epoch_log[NITERATIONS] = 0 # processed images batch counting (may not coincide with the number of optimization steps)

        self.log = {LOSS : list()}
        for name in self.metric_funct_dict.keys():
            self.log[name] = list() # its length is the number of finished epochs

        self.size_dataset = size_dataset
        self.prefix = prefix or ''
    
    def new_epoch(self):
        for key in self.epoch_log.keys():
            self.epoch_log[key] = 0

    def update_epoch_log(self, output, targets, loss, VERBOSE=True):
        for name, funct in self.metric_funct_dict.items():
            self.epoch_log[name] += funct(output, targets)
        self.epoch_log[LOSS] += loss
        self.epoch_log[NTOKENS] += targets.shape[0] # batch_size
        self.epoch_log[NITERATIONS] += 1

        if VERBOSE == True:
            if self.epoch_log[NITERATIONS] == 2 or self.epoch_log[NITERATIONS] == 5 or self.epoch_log[NITERATIONS] % 10 == 0:
                metric0_name = list(self.metric_funct_dict.keys())[0]
                metric0 = 100 * self.epoch_log[metric0_name] / self.epoch_log[NTOKENS]
                loss = self.epoch_log[LOSS] / self.epoch_log[NTOKENS]

                if isinstance(self.size_dataset, int):
                    epoch_completion = 100 * self.epoch_log[NTOKENS] / self.size_dataset # If sizr dataset => NTOKENS / size; If size data loader => NITERATIONS / size
                    print(f"{self.prefix} [epoch_{len(self.log[LOSS])}: {epoch_completion:.1f}%]: num_updates={self.epoch_log[NITERATIONS]}, {metric0_name}={metric0:.1f}, loss={loss:.2f}")
                else:
                    print(f"{self.prefix} [epoch_{len(self.log[LOSS])}: ???%] num_updates={self.epoch_log[NITERATIONS]}, {metric0_name}={metric0:.1f}, loss={loss:.2f}")

    def finish_epoch(self, VERBOSE=True):
        loss = self.epoch_log[LOSS] / self.epoch_log[NTOKENS]
        self.log[LOSS].append(loss)

        for name in self.metric_funct_dict.keys():
            value = 100 * self.epoch_log[name] / self.epoch_log[NTOKENS]
            self.log[name].append(value)
        
        if VERBOSE == True:
            metric0_name = list(self.metric_funct_dict.keys())[0]
            metric0 = self.log[metric0_name][-1]

            print(f"{self.prefix} [epoch_{len(self.log[LOSS]) - 1}: 100%]: num_updates={self.epoch_log[NITERATIONS]}, {metric0_name}={metric0:.1f}, loss={loss:.2f}")

    def best_epochs(self, key=None, num_elems=1, offset=0, maximize=True):
        key = key or list(self.metric_funct_dict.keys())[0]
        vector = self.log[key].copy()
        epochs = np.argsort(vector) + offset
        if maximize : epochs = epochs[::-1]
        
        return epochs[:num_elems].tolist()

    def get_one_epoch_log(self, epoch, new_prefix=None):
        if new_prefix is not None:
            prefix = new_prefix
        else:
            prefix = f'{self.prefix}_' if self.prefix != '' else ''

        return {f'{prefix}{k}' : v[epoch] for k, v in self.log.items()}

    def get_last_epoch_log(self, new_prefix=None):
        return self.get_one_epoch_log(epoch=-1, new_prefix=new_prefix)

    def print_last_epoch_summary(self, mode, extra_notes=None):
        # mode like "train", "valid", "test", etc
        epoch = len(self.log[LOSS])

        metrics_str = ''
        for name in self.metric_funct_dict.keys():
            metrics_str = f"{metrics_str}{mode} {name}={self.log[name][-1]:.1f}%, "
        
        loss = self.log[LOSS][-1]

        if extra_notes is None:
            print(f'\t| epoch {epoch:03d} | {metrics_str}{mode} loss={loss:.2f} |')
        else:
            print(f'\t| epoch {epoch:03d} | {metrics_str}{mode} loss={loss:.2f} ({extra_notes}) |')
