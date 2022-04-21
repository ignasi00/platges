
from .training.accumulated_grad_train import accumulated_grad_train
from .training.vanilla_train import vanilla_train
from .validation.vanilla_validate import vanilla_validate


def kfolds_manager(
    epochs_manager,
    num_epochs,
    train_dataset_iterator,
    val_dataset_iterator,
    create_dataloader,
    model_iterator,
    loss_type,
    optim_type,
    kfolds_logger,
    wandb_logger,
    postprocess_output_and_target_funct=None,
    max_batch_size=1
    ):

    for train_dataset, val_dataset, model in zip(train_dataset_iterator, val_dataset_iterator, model_iterator):
        
        train_dataloader = create_dataloader(train_dataset)
        batch_size = train_dataloader.batch_size
        val_dataloader = create_dataloader(val_dataset)
        
        # model already build
        criterion = loss_type(model)
        optim = optim_type(model)

        train_local_logger, val_local_logger = kfolds_logger.new_fold(train_len_dataset=len(train_dataset), val_len_dataset=len(val_dataset), fold_sufix=True)

        experiment_fold_name = f'{wandb_logger.get_base_experiment_name()}_{kfolds_logger.get_current_fold()}-{kfolds_logger.get_num_folds()}'
        wandb_logger.change_experiment_name(experiment_fold_name)

        # wandb_logger.watch_model(model, log="all", log_freq=50) <- Big models weights too much

        if min(batch_size, max_batch_size) == batch_size:
            train_rutine = lambda : vanilla_train(model, criterion, optim, train_dataloader, train_local_logger, postprocess_output_and_target_funct=postprocess_output_and_target_funct, VERBOSE_BATCH=True, VERBOSE_END=True)
        else:
            train_rutine = lambda : accumulated_grad_train(model, criterion, optim, train_dataloader, train_local_logger, postprocess_output_and_target_funct=postprocess_output_and_target_funct, drop_last=True, VERBOSE_BATCH=True, VERBOSE_END=True)
        val_rutine = lambda : vanilla_validate(model, criterion, val_dataloader, val_local_logger, postprocess_output_and_target_funct=postprocess_output_and_target_funct, VERBOSE_BATCH=True, VERBOSE_END=True)
        
        epochs_manager(model, train_rutine, val_rutine, num_epochs, train_local_logger, val_local_logger, wandb_logger)

        best_epoch = kfolds_logger.finish_fold()

        wandb_logger.summarize({f'best_epoch (fold {kfolds_logger.get_current_fold()}/{kfolds_logger.get_num_folds()})' : best_epoch})
        wandb_logger.summarize(train_local_logger.get_one_epoch_log(best_epoch, prefix=f"train_{kfolds_logger.get_current_fold()}-{kfolds_logger.get_num_folds()}_"))
        wandb_logger.summarize(val_local_logger.get_one_epoch_log(best_epoch, prefix=f"valid_{kfolds_logger.get_current_fold()}-{kfolds_logger.get_num_folds()}_"))
    
    v_best_epoch = kfolds_logger.get_v_best_epoch()
    wandb_logger.summarize({'best_epoch' : max(v_best_epoch)})
