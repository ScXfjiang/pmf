import argparse

import matplotlib.pyplot as plt
import torch

from dataset import MovieLens100K
from dataset import NetflixPrize
from dataset import Yelp
from model import ProbabilisticMatrixFactorization
from util import time_convert


class Trainer(object):
    def __init__(self, model, train_loader, num_epoch, optimizer, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.test_loader = test_loader
        self.train_loss_list = []
        self.test_loss_list = []

    def train(self):
        for epoch_idx in range(1, self.num_epoch + 1):
            print("train epoch {}".format(epoch_idx))
            self.train_epoch()
            print("test epoch {}".format(epoch_idx))
            self.test_epoch()
        plt.plot(self.train_loss_list)
        plt.show()
        plt.plot(self.test_loss_list)
        plt.show()

    def train_epoch(self):
        self.model.train()
        cur_losses = []
        for batch_idx, (user_indices, item_indices, gt_ratings) in enumerate(
            self.train_loader
        ):
            user_indices = user_indices.to(torch.device("cpu"), dtype=torch.int32)
            item_indices = item_indices.to(torch.device("cpu"), dtype=torch.int32)
            gt_ratings = gt_ratings.to(torch.device("cpu"), dtype=torch.float32)
            self.optimizer.zero_grad()
            estimate_ratings = self.model(user_indices, item_indices)
            loss = torch.sqrt(
                torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
            )
            cur_losses.append(loss)
            loss.backward()
            self.optimizer.step()
        self.train_loss_list.append(float(sum(cur_losses) / len(cur_losses)))

    def test_epoch(self):
        self.model.eval()
        cur_losses = []
        for batch_idx, (user_indices, item_indices, gt_ratings) in enumerate(
            self.test_loader
        ):
            user_indices = user_indices.to(torch.device("cpu"), dtype=torch.int32)
            item_indices = item_indices.to(torch.device("cpu"), dtype=torch.int32)
            gt_ratings = gt_ratings.to(torch.device("cpu"), dtype=torch.float32)
            estimate_ratings = self.model(user_indices, item_indices)
            loss = torch.sqrt(
                torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
            )
            cur_losses.append(loss)
        self.test_loss_list.append(float(sum(cur_losses) / len(cur_losses)))


def main():
    parser = argparse.ArgumentParser(description="Probabilistic Matrix Factorization")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--latent_dim", type=int, default=20)
    args = parser.parse_args()

    dataset = Yelp(args.dataset)
    train_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_set_size, test_set_size]
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.test_batch_size,
        shuffle=args.shuffle,
        drop_last=True,
    )
    model = ProbabilisticMatrixFactorization(
        dataset.get_num_user(), dataset.get_num_item(), args.latent_dim
    )
    for param in model.parameters():
        print(type(param.data), param.size())
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    trainer = Trainer(model, train_loader, args.num_epoch, optimizer, test_loader)
    trainer.train()


if __name__ == "__main__":
    main()
