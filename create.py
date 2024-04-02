import numpy as np

def create_dataset(dataset, step):
    X, Y = [], []
    for i in range(len(dataset) - step - 1):
        a = dataset[i:(i + step), 0]
        X.append(a)
        Y.append(dataset[i + step, 0])
    return np.array(X), np.array(Y)