
# remember a last epoch log to commit it.
# remember to wait for the last artifact to upload before downloading the best

import wandb


class WandbLogger():

    def __init__(self, project_name, experiment_name, entity):
        # wandb.init creates (and return) a wandb.Run object available on wandb.run
        # only 1 run should exist at the same script at the same time...
        wandb.init(project=f"{project_name}", entity=entity)

        self.entity = entity
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.base_experiment_name = experiment_name

    def get_base_experiment_name(self) : return self.base_experiment_name
    def get_experiment_name(self) : return self.experiment_name

    def change_experiment_name(self, experiment_name):
        # By instance, for K-folds
        self.experiment_name = experiment_name

    def watch_model(self, model, log="all", log_freq=1000):
        # watch a given run
        wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, dict_data, step=None, commit=None, prefix=None):
        if prefix : dict_data = { f'{prefix}{k}' : v for k, v in dict_data.items()}
        wandb.log(dict_data, step=step, commit=commit)

    def upload_model(self, model_file, aliases=None, wait=True):
        model_io = wandb.Artifact(self.experiment_name, type="model_parameters")
        model_io.add_file(model_file)

        aliases = aliases or []
        aliases = ['latest'] + aliases

        wandb.log_artifact(model_io, aliases=aliases)
        if wait: model_io.wait()

    def update_model(self, search_alia, new_alias_list):
        model_io = wandb.run.use_artifact(f'{self.entity}/{self.project_name}/{self.experiment_name}:{search_alia}', type="model_parameters")
        for alia in new_alias_list:
            model_io.aliases.append(alia)
        model_io.save()

    def upload_submission(self, submission_file, aliases=None, wait=False):
        submission_io = wandb.Artifact(f'{self.experiment_name}_submission', type="submissions")
        submission_io.add_file(submission_file)

        aliases = aliases or []
        aliases = ['latest'] + aliases

        wandb.log_artifact(submission_io, aliases=aliases)
        if wait: submission_io.wait()
    
    def summarize(self, dict_data):
        # log on a given run single final values like best scores (overwritting), configs, best epoch, etc
        #for key, value in dict_data.items():
        #    wandb.run.summary[key] = value
        wandb.run.summary.update(dict_data)

    def download_model(self, model_filename, output_dir, alias=None):
        alias = alias or "latest"
        # Query W&B for an artifact and mark it as input to this run
        artifact = wandb.run.use_artifact(f'{self.entity}/{self.project_name}/{self.experiment_name}:{alias}', type="model_parameters")

        # Download the artifact's contents
        artifact_dir = artifact.download()

        return f"{output_dir}/{model_filename}"

    def cleanup_cache(self, limit=5):
        cache = wandb.sdk.interface.artifacts.get_artifacts_cache()
        cache.cleanup(wandb.util.from_human_size(f"{limit}GB"))
