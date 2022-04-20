

def vanilla_train(model, criterion, optimizer, dataloader, local_logger, postprocess_output_and_target_funct=None, device=None, *, VERBOSE_BATCH=False, VERBOSE_END=False):
    device = device # or torch.device('cpu')

    model.train()
    local_logger.new_epoch()

    for input_, target in dataloader:
        # Dataloader should send its output to the requiered device and set the dtype (see collate_fn)
        if device is not None:
            input_.to(device)
            target.to(device)

        model.zero_grad()

        output = model(input_)
        if postprocess_output_and_target_funct is not None:
            output, target = postprocess_output_and_target_funct(output, target)
            
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() # TODO: rutine with lr scheduler

        local_logger.update_epoch_log(output, target, loss, VERBOSE=VERBOSE_BATCH)

    local_logger.finish_epoch(VERBOSE=VERBOSE_END)
    return local_logger.get_last_epoch_log()
