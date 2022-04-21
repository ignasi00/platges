
import torch


def vanilla_validate(model, criterion, dataloader, local_logger, postprocess_output_and_target_funct=None, device=None, model_eval=True, *, new_prefix=None, VERBOSE_BATCH=False, VERBOSE_END=False):
    new_prefix = new_prefix or ''
    device = device # or torch.device('cpu')
    
    if model_eval : model.eval()
    local_logger.new_epoch()

    with torch.no_grad():
        for input_, target in dataloader:
            # Dataloader could send its output to the requiered device and set the dtype (see collate_fn)
            if device is not None:
                input_.to(device)
                target.to(device)

            output = model(input_)
            if postprocess_output_and_target_funct is not None:
                output, target = postprocess_output_and_target_funct(output, target)
                
            loss = criterion(output, target)

            local_logger.update_epoch_log(output, target, loss, VERBOSE=VERBOSE_BATCH)

    local_logger.finish_epoch(VERBOSE=VERBOSE_END)
    return local_logger.get_last_epoch_log(new_prefix=new_prefix)
