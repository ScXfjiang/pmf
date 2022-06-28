import os

import torch


class MovieLens100K(torch.utils.data.Dataset):
    """
    # https://grouplens.org/datasets/movielens/
    """

    def __init__(self, dataset_path):
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


if __name__ == "__main__":
    ml_100k = MovieLens100K("/Users/xfjiang/workspace/dataset/ml-100k")
    print("num ratings: {}".format(len(ml_100k)))
    print("num users: {}".format(ml_100k.get_num_user()))
    print("num items: {}".format(ml_100k.get_num_item()))
