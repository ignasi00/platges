
from framework.pytorch.datasets.utils.build_kfolds_datasets_generators import build_kfolds_datasets_generators

from .augmented_datasets import build_train_dataset, build_val_dataset
from .concat_dataset import build_concat_dataset


def build_kfolds_datasets_iterators(lists_paths, folds, base_dataset_type, params):
    # Folds is a set of lists of the indices of lists_paths used for the validation set (the complementary is for training)

    def build_train_dataset_from_lists_paths(train_lists):
        base_dataset = build_concat_dataset(train_lists, base_dataset_type)
        train_dataset = build_train_dataset(base_dataset, params.resize_height, params.resize_width, params.crop_height, params.crop_width, params.mean, params.std, params.scale_limit, params.shift_limit, params.rotate_limit)
        return train_dataset

    def build_val_dataset_from_lists_paths(val_lists):
        base_dataset = build_concat_dataset(val_lists, base_dataset_type)
        val_dataset = build_val_dataset(base_dataset, params.resize_height, params.resize_width, params.crop_height, params.crop_width, params.mean, params.std)
        return val_dataset

    train_dataset_generator, val_dataset_generator = build_kfolds_datasets_generators(lists_paths, folds, build_train_dataset_from_lists_paths, build_val_dataset_from_lists_paths)
    
    train_dataset_iterator = train_dataset_generator()
    val_dataset_iterator = val_dataset_generator()

    return train_dataset_iterator, val_dataset_iterator
