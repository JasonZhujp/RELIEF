import torch
import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold


# splitter function
def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold


def scaffold_split_fewshot(dataset, smiles_list, task_idx=None, null_value=0, number_train=100, frac_valid=0.1, frac_test=0.1, seed=0):
    frac_train = 1.0 - frac_valid - frac_test
    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    random.seed(seed)
    random.shuffle(train_idx)
    # determine number of training graphs by ratio (used for examining data efficiency)
    if 0 < number_train <= 1:
        number_train_ratio = number_train * 0.8  # shots percentage is corresponding to the training dataset, which is 80% of total dataset.
        train_dataset = dataset[torch.tensor(train_idx)[:int(number_train_ratio * len(dataset))]]
    # determine number of training graphs directly by number (used for few-shot graph classification)
    else:
        train_dataset = dataset[torch.tensor(train_idx)[:int(number_train)]]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset
    