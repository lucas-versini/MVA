"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    X_train = np.zeros((n_train, max_train_card))
    y_train = np.zeros(n_train)
    for i in range(n_train):
        card = np.random.randint(1, max_train_card + 1)
        multiset = np.random.randint(1, max_train_card + 1, size = card)

        X_train[i, -card:] = multiset
        y_train[i] = multiset.sum()
    ##################

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    X_test, y_test = list(), list()
    for card in range(5, 101, 5):
        multiset = np.random.randint(1, 11, size = (1000, card))
        X_test.append(multiset)
        y_test.append(multiset.sum(axis = 1))
    ##################

    return X_test, y_test
