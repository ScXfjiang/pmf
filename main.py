import argparse

import matplotlib.pyplot as plt
import torch
import time

from dataset import MovieLens100K
from dataset import NetflixPrize
from dataset import Yelp
from model import ProbabilisticMatrixFactorization
from util import show_elapsed_time


class Trainer(object):
    def __init__(
        self, model, train_loader, num_epoch, optimizer, test_loader, use_cuda, K
    ):
        self.model = model
        self.train_loader = train_loader
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.test_loader = test_loader
        self.use_cuda = use_cuda
        self.K = K
        self.train_loss_list = []
        self.test_loss_list = []

    def train(self):
        for epoch_idx in range(1, self.num_epoch + 1):
            train_epoch_start = time.time()
            self.train_epoch()
            train_epoch_end = time.time()
            show_elapsed_time(
                train_epoch_start, train_epoch_end, "Train epoch {}".format(epoch_idx)
            )
            test_epoch_start = time.time()
            self.test_epoch()
            test_epoch_end = time.time()
            show_elapsed_time(
                test_epoch_start, test_epoch_end, "Test epoch {}".format(epoch_idx)
            )

        plt.plot(self.train_loss_list, label="train", linewidth=1)
        plt.plot(self.test_loss_list, label="test", linewidth=1)
        plt.xlabel("num of epoch")
        plt.ylabel("RMSE")
        plt.grid()
        plt.savefig("rmse_curve.pdf")

    def train_epoch(self):
        self.model.train()
        if torch.cuda.is_available() and self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        cur_losses = []
        for batch_idx, (user_indices, item_indices, gt_ratings) in enumerate(
            self.train_loader
        ):
            if torch.cuda.is_available() and self.use_cuda:
                user_indices = user_indices.to(device="cuda")
                item_indices = item_indices.to(device="cuda")
                gt_ratings = gt_ratings.to(device="cuda")
            self.optimizer.zero_grad()
            estimate_ratings = torch.sigmoid(self.model(user_indices, item_indices))
            gt_ratings = (gt_ratings.to(torch.float32) - 1) / (self.K - 1)
            mse = torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
            rmse = loss = torch.sqrt(mse)
            cur_losses.append(rmse)
            rmse.backward()
            self.optimizer.step()
        self.train_loss_list.append(float(sum(cur_losses) / len(cur_losses)))

    def test_epoch(self):
        self.model.eval()
        if torch.cuda.is_available() and self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        cur_losses = []
        for batch_idx, (user_indices, item_indices, gt_ratings) in enumerate(
            self.test_loader
        ):
            if torch.cuda.is_available() and self.use_cuda:
                user_indices = user_indices.to(device="cuda")
                item_indices = item_indices.to(device="cuda")
                gt_ratings = gt_ratings.to(device="cuda")
            estimate_ratings = torch.sigmoid(self.model(user_indices, item_indices))
            gt_ratings = (gt_ratings.to(torch.float32) - 1) / (self.K - 1)
            mse = torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
            rmse = torch.sqrt(mse)
            cur_losses.append(rmse)
        self.test_loss_list.append(float(sum(cur_losses) / len(cur_losses)))


def main():
    parser = argparse.ArgumentParser(description="Probabilistic Matrix Factorization")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--use_cuda", type=bool, default=True)
    args = parser.parse_args()

    print("Dataset initialization starts")
    dataset_init_start = time.time()
    dataset = Yelp(args.dataset)
    dataset_init_end = time.time()
    show_elapsed_time(dataset_init_start, dataset_init_end, "Dataset initialization")
    print("Dataset initialization ends")

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
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    trainer = Trainer(
        model,
        train_loader,
        args.num_epoch,
        optimizer,
        test_loader,
        args.use_cuda,
        args.K,
    )
    trainer.train()


if __name__ == "__main__":
    main()
