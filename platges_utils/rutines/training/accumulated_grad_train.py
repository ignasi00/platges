

def accumulated_grad_train(model, criterion, optimizer, dataloader, local_logger, batch_size, postprocess_output_and_target_funct=None, device=None, drop_last=True, *, new_prefix=None, VERBOSE_BATCH=False, VERBOSE_END=False):
    new_prefix = new_prefix or ''
    device = device # or torch.device('cpu')

    model.train()
    local_logger.new_epoch()
    processed_samples = 0

    model.zero_grad()
    for input_, target in dataloader:
        # Dataloader could send its output to the requiered device and set the dtype (see collate_fn)
        if device is not None:
            input_.to(device)
            target.to(device)

        output = model(input_)
        if postprocess_output_and_target_funct is not None:
            output, target = postprocess_output_and_target_funct(output, target)
            
        loss = criterion(output, target)
        loss.backward()

        processed_samples += target.numel()
        if processed_samples >= batch_size:
            processed_samples = 0
            optimizer.step() # TODO: rutine with lr scheduler

        local_logger.update_epoch_log(output, target, loss, VERBOSE=VERBOSE_BATCH)

    if (not drop_last) and (processed_samples > 0):
        optimizer.step()

    local_logger.finish_epoch(VERBOSE=VERBOSE_END)
    return local_logger.get_last_epoch_log(new_prefix=new_prefix)
