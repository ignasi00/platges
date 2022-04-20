
import contextlib

from framework.pytorch.loggers.wandb_logger import WandbLogger


class WandbFinalSummarize(object):
    # TODO: Diferents versions, one for kfolds, one for others, etc (having it like this is very UGLY)
    def __init__(self, training_type, local_logger_list, best_epochs_funct=None, best_epoch_offset=0):
        """
        local_logger_list for vanilla is element 0 for train and element 1 for validation
        best_epoch_funct is a callable that say what epoch is the best:
            usually lambda : val_local_logger.best_epochs()[0] for vanilla training
            usually lambda : kfolds_local_logger.best_summaries() that returns (v_best_train, v_best_valid, save_key_train, save_key_val) for kfolds 
        """

        self.training_type = training_type
        try:
            len(local_logger_list)
            self.local_logger_list = local_logger_list
        except:
            self.local_logger_list = [local_logger_list]
        
        default_best = lambda : self.local_logger_list[-1].best_epochs()[0]
        self.best_epochs_funct = best_epochs_funct or default_best
        self.best_epoch_offset = best_epoch_offset
    
    def __call__(self, wandb_logger):
        if self.training_type == "kfolds":
            v_best_train, v_best_valid, save_key_train, save_key_val = best_epochs_funct()
            wandb_logger.summarize({
                f'{save_key_train}' : sum([elem[f'{save_key_train}'] for elem in v_best_train]) / len(v_best_train),
                f'{save_key_val}' : sum([elem[f'{save_key_val}'] for elem in v_best_valid]) / len(v_best_valid)
            })

        elif self.training_type in ["vanilla"]:
            best_epoch = best_epoch_funct()
            wandb_logger.summarize({'best_epoch' : self.best_epoch_offset + best_epoch})
            wandb_logger.summarize(self.local_logger_list[0].get_one_epoch_log(best_epoch, prefix="train_"))
            wandb_logger.summarize(self.local_logger_list[1].get_one_epoch_log(best_epoch, prefix="valid_"))
            for i, local_logger in enumerate(self.local_logger_list[2:]):
                wandb_logger.summarize(local_logger.get_one_epoch_log(best_epoch, prefix=f"extra{i + 2}_"))
                
        else:
            raise Exception(f"Undefined training_type: {self.training_type}\nMaybe it is defined but not contemplated on the WandbFinalSummarize class.")


@contextlib.contextmanager
def build_wandb_logger(project_name, experiment_name, entity, exit_funct=None):
    """
    exit_funct should be None or a callable that uses a WandbLogger
    If an object, other inputs should be defined at the constructor
    """
    wandb_logger = WandbLogger(project_name, experiment_name, entity)
    try:
        yield wandb_logger
    finally:
        if exit_funct is not None : exit_funct(wandb_logger)
