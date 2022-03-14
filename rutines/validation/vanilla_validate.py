
def vanilla_validate(model, criterion, dataloader, local_logger, device=None, *, VERBOSE_BATCH=False, VERBOSE_END=False):
    device = device # or torch.device('cpu')
    
    #model.eval()
    local_logger.new_epoch()

    with torch.no_grad():
        for input_, target in dataloader:
            # Dataloader could send its output to the requiered device and set the dtype (see collate_fn)
            if device is not None:
                input_.to(device)
                target.to(device)

            output = model(input_)
            loss = criterion(output, target)

            local_logger.update_epoch_log(output, target, loss, VERBOSE=VERBOSE_BATCH)

    local_logger.finish_epoch(VERBOSE=VERBOSE_END)
    return local_logger.get_last_epoch_log()
