
def vanilla_infer(model, dataloader, single_output_funct=None, list_outputs_funct=None):
    # single_output_funct = lambda x : torch.max(x, 1)[1].detach().to('cpu').numpy()
    # list_outputs_funct = lambda x : np.concatenate(x)
    #model.eval()

    outputs = []
    with torch.no_grad():
        for input_ in dataloader:
            # Dataloader should send its output to the requiered device and set the dtype (see collate_fn)

            output = model(input_)
            if output_funct is not None : output = single_output_funct(output)
            outputs.append(output)

    if list_outputs_funct is not None : outputs = list_outputs_funct(outputs)
    return outputs
