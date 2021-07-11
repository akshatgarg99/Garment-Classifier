import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from train import train
from dataloader import test_loader


if __name__ == "__main__":
    path = "/content/classification-assignment"
    model = train(11, path)
    dataset = test_loader(path)
    dataloader = DataLoader(dataset, batch_size=1)
    filename = []
    neck = []
    sleeve_length = []
    pattern = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch_idx, (input, input_name) in enumerate(dataloader):
            input = input.to(device)
            outputs = model(input)
            filename.append(input_name[0])
            for i in range(3):
                _, predicted = torch.max(outputs[i].data, 1)
                if i == 0:
                    neck.append(predicted[0].item())
                elif i == 1:
                    sleeve_length.append(predicted[0].item())
                else:
                    pattern.append(predicted[0].item())

    model.train()
    output = {"filename": filename, "neck": neck, "sleeve_length": sleeve_length, "pattern": pattern}
    output = pd.DataFrame.from_dict(output)
    output["neck"].replace(7, np.nan, inplace=True)
    output["sleeve_length"].replace(4, np.nan, inplace=True)
    output["pattern"].replace(10, np.nan, inplace=True)
    output.to_csv("Output.csv")
