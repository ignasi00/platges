
import torch


def default_epochs(model, train_rutine, val_rutine, num_epochs, train_local_logger, val_local_logger, wandb_logger, *, epoch_offset=0, models_path='./model_parameters/model.pt'):
    for epoch in range(epoch_offset, num_epochs + epoch_offset):

        last_train_epoch_log = train_rutine()
        wandb_logger.log(last_train_epoch_log, prefix="train_", commit=False)
        train_local_logger.print_last_epoch_summary(mode='train')

        with torch.no_grad():
            last_val_epoch_log = val_rutine()
            wandb_logger.log(last_val_epoch_log, prefix="valid_", commit=False)
            val_local_logger.print_last_epoch_summary(mode='valid')

         # clean up cache
        wandb_logger.cleanup_cache(5)

        # Save model and upload it
        torch.save(model.state_dict(), models_path)
        wait = epoch == (epoch_offset + num_epochs - 1)
        wandb_logger.upload_model(models_path, aliases=[f'epoch_{epoch}'], wait=wait)

        # commit all epoch log
        wandb_logger.log({'epoch' : epoch}, commit=True)
