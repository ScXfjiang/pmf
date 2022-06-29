import os
import re

import torch


class DatasetInterface(torch.utils.data.Dataset):
    def __init__(self):
        super(DatasetInterface, self).__init__()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_num_user(self):
        raise NotImplementedError

    def get_num_item(self):
        raise NotImplementedError


class MovieLens100K(DatasetInterface):
    """
    # https://grouplens.org/datasets/movielens/
    """

    def __init__(self, dataset_path):
        super(MovieLens100K, self).__init__()
        self.structured_data = []
        with open(os.path.join(dataset_path, "u.data")) as f:
            for line in f:
                (user_id, item_id, rating, _) = line.split("\t")
                self.structured_data.append([int(user_id), int(item_id), int(rating)])
        with open(os.path.join(dataset_path, "u.info")) as f:
            u_info = f.readlines()
            self.num_user = int(u_info[0].split(" ")[0])
            self.num_item = int(u_info[1].split(" ")[0])

    def __getitem__(self, idx):
        user_idx = self.structured_data[idx][0] - 1
        item_idx = self.structured_data[idx][1] - 1
        rating = self.structured_data[idx][2]
        return [user_idx, item_idx, rating]

    def __len__(self):
        return len(self.structured_data)

    def get_num_user(self):
        return self.num_user

    def get_num_item(self):
        return self.num_item


class NetflixPrize(DatasetInterface):
    def __init__(self, dataset_path):
        super(NetflixPrize, self).__init__()
        self.structured_data = []
        self.user_ids = []
        self.movie_ids = []
        self.user_id2user_idx = {}
        self.movie_id2movie_idx = {}
        num_file = 4
        movie_id = 0
        for i in range(1, num_file + 1):
            with open(
                    os.path.join(dataset_path, "combined_data_{}.txt".format(i))
            ) as f:
                for line in f:
                    line = line.strip()
                    if re.match("[0-9]+:", line):
                        movie_id = int(line[:-1])
                        self.movie_ids.append(movie_id)
                        continue
                    else:
                        (user_id, rating, _) = line.split(",")
                        if user_id not in self.user_ids:
                            self.user_ids.append(user_id)
                        self.structured_data.append(
                            [int(user_id), movie_id, int(rating)]
                        )
        self.user_id2user_idx = {
            user_id: idx for idx, user_id in enumerate(self.user_ids)
        }
        self.movie_id2movie_idx = {
            movie_id: idx for idx, movie_id in enumerate(self.movie_ids)
        }

    def __getitem__(self, idx):
        user_idx = self.user_id2user_idx(self.structured_data[idx][0])
        movie_idx = self.movie_id2movie_idx(self.structured_data[idx][1])
        rating = self.structured_data[idx][2]
        return [user_idx, movie_idx, rating]

    def __len__(self):
        return len(self.structured_data)

    def get_num_user(self):
        return len(self.user_ids)

    def get_num_item(self):
        return len(self.movie_ids)


class Yelp(DatasetInterface):
    pass


if __name__ == "__main__":
    # MovieLens100k
    # ml_100k = MovieLens100K("/Users/xfjiang/workspace/dataset/ml-100k")
    # print("num ratings: {}".format(len(ml_100k)))
    # print("num users: {}".format(ml_100k.get_num_user()))
    # print("num items: {}".format(ml_100k.get_num_item()))

    # Netflix Prize
    netflix_prize = NetflixPrize("/Users/xfjiang/workspace/dataset/netflix_prize")
    print(len(netflix_prize))
    print(netflix_prize.get_num_user())
    print(netflix_prize.get_num_item())
