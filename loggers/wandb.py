
import wandb


class WandB_logger():

    def __init__(self, project_name, experiment_name, entity):
        # start a new run of project_name
        
        # wandb.init creates (and return) a wandb.Run object available on wandb.run
        # only 1 run should exist at the same script at the same time...
        wandb.init(project=f"{project_name}", entity=entity)

        self.experiment_name = experiment_name

    def watch_model(self, model, log="all", log_freq=10):
        # watch a given run
        wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, dict_data, step=None, commit=None):
        # log on a given run over different steps (including wandb.Images, but not lists)
        # it may be useful to log different x-axis instead of using step
        wandb.log(dict_data, step=step, commit=commit)

    def summary(self, dict_data):
        # log on a given run single (final) values like best scores (overwritting)
        # TODO: it is possible to define metric goals (min, max, etc) that will become, individualy, the summary value authomaticaly
        for key, value in dict_data.items():
            wandb.run.summary[key] = value

    def save_model(self, model_file, aliases=None):
        # Globaly save the current model on the experiment_name entry of "model_parameters"
        # TODO: currently it will be made latest is the best at main, see how to manage aliases correctly

        model_io = wandb.Artifact(self.experiment_name, type="model_parameters")
        #wandb.use_artifact(self.model_io)
        
        model_io.add_file(model_file)

        aliases = aliases or []
        aliases = ['latest'] + aliases

        wandb.log_artifact(self.model_io, aliases=aliases)
