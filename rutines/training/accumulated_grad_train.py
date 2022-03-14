

def accumulated_grad_train(model, criterion, optimizer, dataloader, local_logger, batch_size, device=None, drop_last=True, *, VERBOSE_BATCH=False, VERBOSE_END=False):
    device = device or torch.device('cpu')

    #model.train()
    local_logger.new_epoch()
    processed_samples = 0

    model.zero_grad()
    for batch_idx, (input_, target) in enumerate(dataloader):
        # Dataloader could send its output to the requiered device and set the dtype (see collate_fn)
        input_.to(device)
        target.to(device)

        output = model(input_)
        loss = criterion(output, target)
        loss.backward()

        processed_samples += target.numel()
        if processed_samples >= batch_size:
            processed_samples = 0
            optimizer.step() # TODO 2: rutine with lr scheduler

        local_logger.update_epoch_log(output, target, loss, VERBOSE=VERBOSE_BATCH)

    if (not drop_last) and (processed_samples > 0):
        optimizer.step()

    local_logger.finish_epoch(VERBOSE=VERBOSE_END)
    return local_logger.get_last_epoch_log()
