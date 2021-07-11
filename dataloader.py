import os
import random
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class Loader(Dataset):
    def __init__(self, path, Train=True):
        # get the path to the dataset
        path = path

        self.image_path = os.path.join(path, "images")

        # get a list of image names
        self.image_name = os.listdir(self.image_path)

        # remove this file which is created in windows
        self.image_name.remove("Thumbs.db")

        # get lables location and read it
        label_data = pd.read_csv(os.path.join(path, "attributes.csv"))
        values = {"neck": 7, "sleeve_length": 4, "pattern": 10}

        # create a new class for nan values
        self.label_data = label_data.fillna(value=values)

        # agmentations, uncomment to add more audmentations
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]),
                                         # transforms.GaussianBlur(3),
                                         # transforms.RandomHorizontalFlip(0.5)
                                         ])

        # train test split
        random.shuffle(self.image_name)
        if Train:
            self.image_name = self.image_name[:1300]
        else:
            self.image_name = self.image_name[1300:]

        super().__init__()

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, x):
        input_name = self.image_name[x]
        input_location = os.path.join(self.image_path, input_name)

        # read the image
        image = cv2.imread(input_location)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 255.0).astype(np.float32)

        # apply transformations
        image = self.trans(image)

        # get the labels for the image
        row = self.label_data.loc[self.label_data['filename'] == input_name]

        # convert to numpy
        row = row.to_numpy()[0, 1:].astype(np.int)

        return image, (row[0], row[1], row[2])


class test_loader(Dataset):
    def __init__(self, path):
        self.image_path = os.path.join(path, "images")

        # get a list of image names
        self.image_name = os.listdir(self.image_path)

        # remove this file which is created in windows
        self.image_name.remove("Thumbs.db")

        random.shuffle(self.image_name)

        self.image_name = self.image_name[:500]
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, x):
        input_name = self.image_name[x]
        input_location = os.path.join(self.image_path, input_name)

        # read the image
        image = cv2.imread(input_location)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image/255.0).astype(np.float32)

        # apply transformations
        image = self.trans(image)

        return image, input_name
