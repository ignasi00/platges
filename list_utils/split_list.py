
import numpy as np
import pandas as pd
import random


def acumulate_probs(probs):
    return [sum(probs[:i]) for i in range(len(probs))] + [1]

def split_list(list_of_lists, probs, verbose=False):
    list_of_lists = list_of_lists.copy()
    random.shuffle(list_of_lists)

    try:
        for p in probs : break
    except:
        probs = [probs]

    probs = acumulate_probs(probs)
    
    splits = []
    len_ = len(list_of_lists)
    for p0, p1 in zip(probs[:-1], probs[1:]):
        if verbose : print(f"{int(len_ * p0)} ---- {int(len_ * p1)}")
        splits.append(list_of_lists[int(len_ * p0): int(len_ * p1)])

    return tuple(splits)

def split_pandas(pandas_table, probs, verbose=False):
    perm = np.random.permutation(len(pandas_table.index))

    try:
        for p in probs : break
    except:
        probs = [probs]
    
    probs = acumulate_probs(probs)

    splits = []
    len_ = len(perm)
    for p0, p1 in zip(probs[:-1], probs[1:]):
        if verbose : print(f"{int(len_ * p0)} ---- {int(len_ * p1)}")
        splits.append(pandas_table.iloc[perm[int(len_ * p0): int(len_ * p1)]])

    return tuple(splits)
