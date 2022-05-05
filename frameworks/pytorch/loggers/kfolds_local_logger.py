
from .local_logger import LocalLogger


class KfoldsLocalLogger():

    def __init__(self, metric_funct_dict, num_folds, key=None, maximize=True, train_prefix=None, val_prefix=None):
        self.metric_funct_dict = metric_funct_dict
        self.key = key or list(metric_funct_dict.keys())[0]
        self.maximize = maximize

        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.train_local_logger = None
        self.val_local_logger = None

        self.num_folds = num_folds
        self.current_fold = 0

        self.v_best_epoch = []
        self.v_best_train = []
        self.v_best_valid = []

    def get_current_fold(self) : return self.current_fold
    def get_num_folds(self) : return self.num_folds
    def get_v_best_epoch(self) : return self.v_best_epoch

    def new_fold(self, train_len_dataset=None, val_len_dataset=None, fold_sufix=True):
        self.current_fold += 1
        if fold_sufix:
            train_prefix = f'{self.train_prefix} (fold {self.current_fold}/{self.num_folds})'
            val_prefix = f'{self.val_prefix} (fold {self.current_fold}/{self.num_folds})'
        else:
            train_prefix = self.train_prefix
            val_prefix = self.val_prefix

        self.train_local_logger = LocalLogger(self.metric_funct_dict.copy(), size_dataset=train_len_dataset, prefix=train_prefix)
        self.val_local_logger = LocalLogger(self.metric_funct_dict.copy(), size_dataset=val_len_dataset, prefix=val_prefix)

        return self.train_local_logger, self.val_local_logger

    def get_current_local_loggers(self):
        return self.train_local_logger, self.val_local_logger

    def finish_fold(self):
        best_epoch = self.val_local_logger.best_epochs(key=self.key, maximize=self.maximize)[0]

        self.v_best_train.append(self.train_local_logger.get_one_epoch_log(best_epoch, new_prefix=f"train_"))
        self.v_best_valid.append(self.val_local_logger.get_one_epoch_log(best_epoch, new_prefix=f"valid_"))

        self.v_best_epoch.append(best_epoch)
        return best_epoch

    def get_results(self):
        return self.v_best_epoch, self.v_best_train, self.v_best_valid, f'train_{self.key}', f'valid_{self.key}'
