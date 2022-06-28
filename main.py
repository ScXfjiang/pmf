import argparse

import torch

from dataset import MovieLens100K
from model import ProbabilisticMatrixFactorization


def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (user_indices, item_indices, gt_ratings) in enumerate(train_loader):
        user_indices = user_indices.to(torch.device("cpu"), dtype=torch.int32)
        item_indices = item_indices.to(torch.device("cpu"), dtype=torch.int32)
        gt_ratings = gt_ratings.to(torch.device("cpu"), dtype=torch.float32)
        optimizer.zero_grad()
        estimate_ratings = model(user_indices, item_indices)
        loss = torch.sqrt(
            torch.nn.functional.mse_loss(
                estimate_ratings, gt_ratings - torch.mean(gt_ratings)
            )
        )
        print(loss)
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (user_indices, item_indices, gt_ratings) in enumerate(test_loader):
            user_indices = user_indices.to(torch.device("cpu"), dtype=torch.int32)
            item_indices = item_indices.to(torch.device("cpu"), dtype=torch.int32)
            gt_ratings = gt_ratings.to(torch.device("cpu"), dtype=torch.float32)
            estimate_ratings = model(user_indices, item_indices)
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
    parser.add_argument("--train_batch_size", type=int, default="1000")
    parser.add_argument("--test_batch_size", type=int, default="1000")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default="10")
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
        ml_100k.get_num_user(), ml_100k.get_num_item(), args.hidden_dim
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    for _ in range(1, args.num_epoch + 1):
        train(model, train_loader, optimizer)
        test(model, test_loader)


if __name__ == "__main__":
    main()
