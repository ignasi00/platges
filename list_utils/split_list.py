
import numpy as np
import pandas as pd
import random


def split_list(list_of_lists, probs):
    list_of_lists = list_of_lists.copy()
    random.shuffle(list_of_lists)

    try:
        for p in probs : break
    except:
        probs = [probs]
    
    splits = []
    len_ = len(list_of_lists)
    for p0, p1 in zip([0] + probs[:-2], probs[:-1]):
        splits.append(list_of_lists[int(len_ * p0): int(len_ * p1)])
    splits.append(list_of_lists[int(len_ * p1):])

    return splits[0], *splits[1:]

def split_pandas(pandas_table, probs):
    perm = np.random.permutation(pandas_table.index)

    try:
        for p in probs : break
    except:
        probs = [probs]
    
    splits = []
    len_ = len(perm)
    for p0, p1 in zip([0] + probs[:-2], probs[:-1]):
        splits.append(pandas_table.iloc[int(len_ * p0): int(len_ * p1)])
    splits.append(pandas_table.iloc[int(len_ * p1):])

    return splits[0], *splits[1:]
