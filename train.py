import torch
import time
from torch.utils.data import DataLoader
import numpy as np

from model import Clf_Model
from loss_function import Loss_function
from dataloader import Loader
from utils import loss_weights


def evaluate(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    total = [0, 0, 0]
    correct = [0, 0, 0]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), [i.to(device) for i in targets]

            outputs = model(inputs)
            for i in range(3):
                _, predicted = torch.max(outputs[i].data, 1)
                total[i] += targets[i].size(0)
                correct[i] += predicted.eq(targets[i].data).cpu().sum()
        ratio = [correct[i] / total[i] for i in range(3)]

        print(ratio)
        print('--------------------------------------------------------------')
    model.train()
    return sum(ratio) / 3


def train(epochs, data_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Clf_Model()
    model = model.to(device)

    train_set = Loader(data_path)
    val_set = Loader(data_path, False)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True, num_workers=4)

    # get class weights for the loss functions
    loss_balance = loss_weights(data_path)

    criterion = Loss_function(loss_balance)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1, verbose=True)
    losses = []

    checkpoint_path = "/content/classification-assignment/model.pt"

    # Train
    start = time.time()

    acu_max = evaluate(model, valloader)

    for epoch in range(epochs):
        print("=======Starting epoch ", epoch, "========")

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), [i.to(device) for i in targets]

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion.total_loss(outputs, targets)
            loss.backward()

            optimizer.step()
            losses.append(loss.item())
            end = time.time()

            if batch_idx % 20 == 0:
                print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_idx, np.mean(losses), end - start))
                print()
                start = time.time()
        scheduler.step()

        # Evaluate
        acu = evaluate(model, valloader)
        if acu > acu_max:
            checkpoint = {'epoch': epoch,
                          'valid_acc_max': acu,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}

            torch.save(checkpoint, checkpoint_path)
            acu_max = acu
            print("----Saving model---")
    return model