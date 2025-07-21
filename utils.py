import copy
from torch import nn


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_model = None
        self.best_loss = float("inf")

    # Returns True when you should stop training, False otherwise.
    def __call__(self, model: nn.Module, val_loss):
        if val_loss < self.best_loss:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            model.load_state_dict(self.best_model)
            return True
