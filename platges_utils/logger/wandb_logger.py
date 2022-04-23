
import contextlib
import numpy as np

from framework.pytorch.loggers.wandb_logger import WandbLogger


class WandbFinalSummarize(object):
    # TODO: Diferents versions, one for kfolds, one for others, etc (having it like this is very UGLY)
    def __init__(self, training_type, local_logger_list, best_epochs_funct=None, best_epoch_offset=0):
        """
        local_logger_list for vanilla is element 0 for train and element 1 for validation
        best_epoch_funct is a callable that say what epoch is the best:
            usually lambda : val_local_logger.best_epochs()[0] for vanilla training
            usually lambda : kfolds_local_logger.best_summaries() that returns (v_best_epoch, v_best_train, v_best_valid, save_key_train, save_key_val) for kfolds 
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
            v_best_epoch, v_best_train, v_best_valid, save_key_train, save_key_val = self.best_epochs_funct()
            wandb_logger.summarize({'best_epoch' : max(v_best_epoch)})
            
            v_valid_metric = [elem[f'{save_key_val}'] for elem in v_best_valid]
            wandb_logger.summarize({
                f'{save_key_train}' : sum([elem[f'{save_key_train}'] for elem in v_best_train]) / len(v_best_train),
                f'{save_key_val}' : sum(v_valid_metric) / len(v_best_valid)
            })

            best_fold = np.argmax(v_valid_metric)
            for fold, epoch in enumerate(v_best_epoch):
                experiment_best_fold_name = f'{wandb_logger.get_base_experiment_name()}_{fold + 1}-{wandb_logger.get_experiment_name().split("-")[-1]}'
                wandb_logger.change_experiment_name(experiment_best_fold_name)

                best_model_alia = f'epoch_{epoch}'
                new_alias = ['best']
                if fold == best_fold : new_alias = new_alias + ['best_kfolds']
                wandb_logger.update_model(best_model_alia, new_alias)

        elif self.training_type in ["vanilla"]:
            best_epoch = self.best_epochs_funct()
            wandb_logger.summarize({'best_epoch' : self.best_epoch_offset + best_epoch})

            wandb_logger.summarize(self.local_logger_list[0].get_one_epoch_log(best_epoch, new_prefix="train_"))
            wandb_logger.summarize(self.local_logger_list[1].get_one_epoch_log(best_epoch, new_prefix="valid_"))
            for i, local_logger in enumerate(self.local_logger_list[2:]):
                wandb_logger.summarize(local_logger.get_one_epoch_log(best_epoch, new_prefix=f"extra{i + 2}_"))
                
            best_model_alia = f"epoch_{best_epoch}"
            wandb_logger.update_model(best_model_alia, ['best'])


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
