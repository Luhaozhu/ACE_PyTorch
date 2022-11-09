import torch
from torch.utils.data import DataLoader, TensorDataset
from captum.concept import Classifier
from captum._utils.models.linear_model import model
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union
import random
from sklearn import linear_model
import numpy as np


class CustomClassifier(Classifier):

    def __init__(self):

        # self.lm = model.SkLearnSGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)
        self.lm = linear_model.SGDClassifier(alpha=0.01, max_iter=1000,tol=1e-3)

    def train_and_eval(self, dataloader,test_split_ratio: float = 0.33):

        inputs = []
        labels = []
        for input, label in dataloader:
            inputs.append(input)
            labels.append(label)

        device = "cpu" if input is None else input.device
        x_train, x_test, y_train, y_test = _train_test_split(
            torch.cat(inputs), torch.cat(labels), test_split=test_split_ratio
        )
        self.lm.device = device
        self.lm.fit(x_train.detach().numpy(),y_train.detach().numpy())

        preds = torch.tensor(self.lm.predict(x_test.detach().numpy()))

        return {'accs': (preds == y_test).float().mean()}

    def weights(self):

        r"""
        This function returns a C x F tensor weights, where
        C is the number of classes and F is the number of features.
        In case of binary classification, C = 2 othewise it is > 2.

        Returns:
            weights (tensor): A torch Tensor with the weights resulting from
                the model training.
        """
        coef = self.lm.coef_[0]
        coef = coef / np.linalg.norm(coef, ord=2)  # 进行归一化
        if len(self.lm.coef_) == 1:
            return torch.tensor([-1 * coef, coef])
        else:
            return torch.tensor(coef)

    def classes(self):
        return self.lm.classes_

def _train_test_split(
    x_list: Tensor, y_list: Tensor, test_split: float = 0.33
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Shuffle
    z_list = list(zip(x_list, y_list))
    random.shuffle(z_list)
    # Split
    test_size = int(test_split * len(z_list))
    z_test, z_train = z_list[:test_size], z_list[test_size:]
    x_test, y_test = zip(*z_test)
    x_train, y_train = zip(*z_train)
    return (
        torch.stack(x_train),
        torch.stack(x_test),
        torch.stack(y_train),
        torch.stack(y_test),
    )
