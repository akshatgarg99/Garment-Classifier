import torch
from torchvision import models


class Clf_Model(torch.nn.Module):
    def __init__(self):
        super(Clf_Model, self).__init__()

        transfer_model = models.vgg11(True)
        modules = list(transfer_model.children())[:-2]
        self.transfer_model = torch.nn.Sequential(*modules)

        # freeze the layers but the model works better without that
        # for param in self.transfer_model.parameters():
        # param.requires_grad = False

        # create a detection head with 3 individual ditections
        self.pattern_clf = torch.nn.Sequential(torch.nn.Linear(32256, 1024),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(1024, 11))
        self.neck_clf = torch.nn.Sequential(torch.nn.Linear(32256, 1024),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(1024, 8))
        self.sleeve_clf = torch.nn.Sequential(torch.nn.Linear(32256, 1024),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(1024, 5))

    def forward(self, x):
        x = self.transfer_model(x)
        # flatten
        x = x.view(x.shape[0], -1)

        neck = self.neck_clf(x)
        sleeve = self.sleeve_clf(x)
        pattern = self.pattern_clf(x)

        return neck, sleeve, pattern
