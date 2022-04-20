

def build_kfolds_datasets_generators(lists_paths, folds, build_train_dataset_from_lists_paths, build_val_dataset_from_lists_paths):
    """
    build_train_dataset_from_lists_paths and build_val_dataset_from_lists_paths could have all constants inside.
    train_dataset_generator and val_dataset_generator optional parameters may be useful if there are no constant inputs.
    Reminder 1: a generator builds an iterator (a = generator(); next(a)).
    Reminder 2: the zip of iterators do not expand them, if they are assimetric, the first to end (raise StopIteration) will end the zip.
    """

    def train_dataset_generator(*args, **kargs):
        train_lists = [lists_paths[indx] for indx in range(len(lists_paths)) if indx not in folds]
        train_dataset = build_train_dataset_from_lists_paths(train_lists, *args, **kargs)
        yield train_dataset

    def val_dataset_generator(*args, **kargs):
        val_lists = [lists_paths[indx] for indx in folds]
        val_dataset = build_val_dataset_from_lists_paths(val_lists, *args, **kargs)
        yield val_dataset

    return train_dataset_generator, val_dataset_generator
    