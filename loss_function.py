import torch
import torch.nn.functional as F


class Loss_function:
    def __init__(self, loss_balance):
        # weights for the detection head in the final loss
        self.pattern_weight = 1
        self.neck_weight = 2
        self.sleeve_weight = 1

        # used to apply weights to every class according to the imbalance in data
        self.loss_balance = loss_balance
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def total_loss(self, predict: list, targets: list):
        total_loss = self.neck_loss(predict[0], targets[0]) + self.sleeve_loss(predict[1], targets[1])+ self.pattern_loss(predict[2], targets[2])
        return total_loss

    def pattern_loss(self, y_pred, y_true):

        # get the weight vector for every class in pattern
        weights = torch.tensor(self.loss_balance["pattern"], dtype=torch.float32).to(self.device)
        loss = F.cross_entropy(y_pred, y_true, weight=weights)
        return self.pattern_weight * loss

    def neck_loss(self, y_pred, y_true):
        # get the weight vector for every class in neck type
        weights = torch.tensor(self.loss_balance["neck"], dtype=torch.float32).to(self.device)
        loss = F.cross_entropy(y_pred, y_true, weight=weights)
        return self.neck_weight * loss

    def sleeve_loss(self, y_pred, y_true):
        # get the weight vector for every class in sleeve type
        weights = torch.tensor(self.loss_balance["sleeve_length"], dtype=torch.float32).to(self.device)
        loss = F.cross_entropy(y_pred, y_true, weight=weights)
        return self.sleeve_weight * loss
