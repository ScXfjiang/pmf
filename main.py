import argparse

import matplotlib.pyplot as plt
import torch

from dataset import MovieLens100K
from model import ProbabilisticMatrixFactorization


class Trainer(object):
    def __init__(self, model, train_loader, num_epoch, optimizer, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.test_loader = test_loader
        self.train_loss_list = []

    def train(self):
        for _ in range(self.num_epoch):
            self.train_epoch()
            # self.test_epoch()
        plt.plot(self.train_loss_list)
        plt.show()

    def train_epoch(self):
        self.model.train()
        for batch_idx, (user_indices, item_indices, gt_ratings) in enumerate(self.train_loader):
            user_indices = user_indices.to(torch.device("cpu"), dtype=torch.int32)
            item_indices = item_indices.to(torch.device("cpu"), dtype=torch.int32)
            gt_ratings = gt_ratings.to(torch.device("cpu"), dtype=torch.float32)
            self.optimizer.zero_grad()
            estimate_ratings = self.model(user_indices, item_indices)
            loss = torch.sqrt(
                torch.nn.functional.mse_loss(
                    estimate_ratings, gt_ratings - torch.mean(gt_ratings)
                )
            )
            if batch_idx % 100 == 0:
                self.train_loss_list.append(loss.cpu().detach().numpy())
            loss.backward()
            self.optimizer.step()

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (user_indices, item_indices, gt_ratings) in enumerate(self.test_loader):
                user_indices = user_indices.to(torch.device("cpu"), dtype=torch.int32)
                item_indices = item_indices.to(torch.device("cpu"), dtype=torch.int32)
                gt_ratings = gt_ratings.to(torch.device("cpu"), dtype=torch.float32)
                estimate_ratings = self.model(user_indices, item_indices)
                loss = torch.sqrt(
                    torch.nn.functional.mse_loss(
                        estimate_ratings, gt_ratings - torch.mean(gt_ratings)
                    )
                )


def main():
    parser = argparse.ArgumentParser(description="Probabilistic Matrix Factorization")
    parser.add_argument(
        "--dataset", type=str, default="/Users/xfjiang/workspace/dataset/ml-100k"
    )
    parser.add_argument("--train_batch_size", type=int, default=1000)
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=10)
    args = parser.parse_args()

    ml_100k = MovieLens100K(args.dataset)
    train_set_size = int(len(ml_100k) * 0.8)
    test_set_size = len(ml_100k) - train_set_size
    train_set, test_set = torch.utils.data.random_split(
        ml_100k, [train_set_size, test_set_size]
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
        ml_100k.get_num_user(), ml_100k.get_num_item(), args.latent_dim
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    trainer = Trainer(model, train_loader, args.num_epoch, optimizer, test_loader)
    trainer.train()


if __name__ == "__main__":
    main()
