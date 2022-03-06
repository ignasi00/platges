

def vanilla_train(model, criterion, optimizer, dataloader, local_logger, *, VERBOSE_BATCH=False, VERBOSE_END=False):
    #model.train()
    local_logger.new_epoch()

    for input_, target in dataloader:
        # Dataloader should send its output to the requiered device and set the dtype (see collate_fn)

        model.zero_grad() # TODO 1: rutine where grad is accumulated to increase an optimization batch_size (not equal to GPU batch_size)
        output = model(input_)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() # TODO 1: same of model.zero_grad(); TODO 2: rutine with lr scheduler

        local_logger.update_epoch_log(output, target, loss, VERBOSE=VERBOSE_BATCH)

    local_logger.finish_epoch(VERBOSE=VERBOSE_END)
    return local_logger.get_last_epoch_log()
