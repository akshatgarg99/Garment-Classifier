import os
import pandas as pd


def loss_weights(path):
    # calculates data imbalance and returns weights for the loss fuction

    # get the labels
    label_path = os.path.join(path, "attributes.csv")
    labels = pd.read_csv(label_path)

    # fill nan values
    values = {"neck": 7, "sleeve_length": 4, "pattern": 10}
    labels = labels.fillna(value=values)

    # calculate the data imbalance
    weights = {}
    for i, v in values.items():
        p = labels[i].value_counts()
        n = []
        for cls in range(v+1):
            n.append(p[cls])
        n = [sum(n)/j for j in n]

        weights[i] = n

    # return weights for loss
    return weights
