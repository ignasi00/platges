
import wandb


class WandB_logger():

    def __init__(self, project_name, experiment_name, entity, first_epoch=1):
        # start a new run of project_name
        
        # wandb.init creates (and return) a wandb.Run object available on wandb.run
        # only 1 run should exist at the same script at the same time...
        wandb.init(project=f"{project_name}", entity=entity)

        self.experiment_name = experiment_name
        
        self.epoch_summary = dict()
        self.best_summary = dict()
        self.best_summary_epochs = dict()

    def watch_model(self, model, log="all", log_freq=10):
        # watch a given run
        wandb.watch(model, log=log, log_freq=log_freq)
    
    def log(self, dict_data, step=None, commit=None):
        # log on a given run over different steps (including wandb.Images, but not lists)
        # it may be useful to log different x-axis instead of using step
        wandb.log(dict_data, step=step, commit=commit)

    def log_metrics(self, dict_data, step=None, commit=None):
        # log on a given run over different steps (including wandb.Images, but not lists)
        # it may be useful to log different x-axis instead of using step
        wandb.log(dict_data, step=step, commit=commit)

        for key, value in dict_data.items():
            try:
                self.epoch_summary[key].append(value)
            except:
                self.epoch_summary[key] = [value]

    def log_epoch(self, epoch, step=None, commit=None):
        epoch_log = dict()
        for key, values in self.epoch_summary.items():
            epoch_log[f'epoch_mean_{key}'] = sum(values) / len(values)
            epoch_log[f'epoch_min_{key}'] = min(values)
            epoch_log[f'epoch_max_{key}'] = max(values)
            try:
                if self.best_summary[key] > epoch_log[f'epoch_mean_{key}']:
                    self.best_summary[key] = epoch_log[f'epoch_mean_{key}']
                    self.best_summary_epochs[f'best_{key}'] = epoch
            except:
                self.best_summary[key] = epoch_log[f'epoch_mean_{key}']
                self.best_summary_epochs[f'best_{key}'] = epoch

        wandb.log({'epoch_idx' : epoch}, setp=step, commit=False)
        wandb.log(epoch_log, step=step, commit=commit)
        
        self.epoch_summary = dict()

        return epoch_log

    def summary(self, dict_data):
        # log on a given run single final values like best scores (overwritting) or config
        for key, value in dict_data.items():
            wandb.run.summary[key] = value
    
    def summary_metrics(self):
        # log on a given run single final values like best scores (overwritting) or config
        for key, value in self.best_summary.items():
            wandb.run.summary[key] = value
        for key, value in self.best_summary_epochs.items():
            wandb.run.summary[f"{key}_epoch"] = value

    def save_model(self, model_file, aliases=None):
        # Globaly save the current model on the experiment_name entry of "model_parameters"
        # TODO: currently it will be made latest is the best at main, see how to manage aliases correctly

        model_io = wandb.Artifact(self.experiment_name, type="model_parameters")
        #wandb.use_artifact(self.model_io)
        
        model_io.add_file(model_file)

        aliases = aliases or []
        aliases = ['latest'] + aliases

        wandb.log_artifact(model_io, aliases=aliases)

    def update_models(self):
        for key, values in self.best_summary_epochs.items():
            model_io = wandb.Api().artifact(f"{self.experiment_name}:epoch_{value}")
            artifact.aliases.append(key)
            artifact.save()
