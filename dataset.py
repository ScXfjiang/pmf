import os
import re

import torch
import json
import pickle as pkl
import time


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
    https://grouplens.org/datasets/movielens/
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
    """
    https://www.yelp.com/dataset/documentation/main/
    """
    def __init__(self, dataset_path):
        print("Dataset initialization starts")
        dataset_init_start = time.time()

        super(Yelp, self).__init__()

        cache_dir = os.path.join(os.getcwd(), "cache")
        if os.path.exists(cache_dir):
            print("Initialize dataset from cache")
            with open(os.path.join(cache_dir, "review_json_list.pkl"), "rb") as f:
                self.review_json_list = pkl.load(f)
            with open(os.path.join(cache_dir, "user_ids.pkl"), "rb") as f:
                self.user_ids = pkl.load(f)
            with open(os.path.join(cache_dir, "business_ids.pkl"), "rb") as f:
                self.business_ids = pkl.load(f)
            with open(os.path.join(cache_dir, "user_id2user_idx.pkl"), "rb") as f:
                self.user_id2user_idx = pkl.load(f)
            with open(os.path.join(cache_dir, "user_idx2user_id.pkl"), "rb") as f:
                self.user_idx2user_id = pkl.load(f)
            with open(
                os.path.join(cache_dir, "business_id2business_idx.pkl"), "rb"
            ) as f:
                self.business_id2business_idx = pkl.load(f)
            with open(
                os.path.join(cache_dir, "business_idx2business_id.pkl"), "rb"
            ) as f:
                self.business_idx2business_id = pkl.load(f)
        else:
            # Some user_ids in review.json are not found in user.json.
            # This issue doesn't happen to business_ids. To be consistent,
            # I build data structures of both users and business from review.json.
            print("Initialize dataset from raw data")
            os.mkdir(cache_dir)
            self.review_json_list = []
            user_ids_set = set()
            business_ids_set = set()
            self.stars = []
            with open(
                os.path.join(dataset_path, "yelp_academic_dataset_review.json"), "rb"
            ) as f:
                for line in f:
                    review_json = json.loads(line)
                    self.review_json_list.append(review_json)
                    self.stars.append(review_json["stars"])
                    user_ids_set.add(review_json["user_id"])
                    business_ids_set.add(review_json["business_id"])
            self.user_ids = list(user_ids_set)
            self.business_ids = list(business_ids_set)
            self.user_id2user_idx = {
                user_id: user_idx for user_idx, user_id in enumerate(self.user_ids)
            }
            self.user_idx2user_id = {
                user_idx: user_id for user_idx, user_id in enumerate(self.user_ids)
            }
            self.business_id2business_idx = {
                business_id: business_idx
                for business_idx, business_id in enumerate(self.business_ids)
            }
            self.business_idx2business_id = {
                business_idx: business_id
                for business_idx, business_id in enumerate(self.business_ids)
            }
            with open(os.path.join(cache_dir, "review_json_list.pkl"), "wb") as f:
                pkl.dump(self.review_json_list, f)
            with open(os.path.join(cache_dir, "user_ids.pkl"), "wb") as f:
                pkl.dump(self.user_ids, f)
            with open(os.path.join(cache_dir, "business_ids.pkl"), "wb") as f:
                pkl.dump(self.business_ids, f)
            with open(os.path.join(cache_dir, "user_id2user_idx.pkl"), "wb") as f:
                pkl.dump(self.user_id2user_idx, f)
            with open(os.path.join(cache_dir, "user_idx2user_id.pkl"), "wb") as f:
                pkl.dump(self.user_idx2user_id, f)
            with open(
                os.path.join(cache_dir, "business_id2business_idx.pkl"), "wb"
            ) as f:
                pkl.dump(self.business_id2business_idx, f)
            with open(
                os.path.join(cache_dir, "business_idx2business_id.pkl"), "wb"
            ) as f:
                pkl.dump(self.business_idx2business_id, f)

        dataset_init_end = time.time()
        second = int(dataset_init_end - dataset_init_start)
        hour = second // 3600
        second = second - hour * 3600
        min = second // 60
        second = second - min * 60
        print(
            "Dataset initialization elapsed time: {} hours {} mins {} seconds".format(
                hour, min, second
            )
        )
        print("Dataset initialization ends")

    def __getitem__(self, idx):
        review_json = self.review_json_list[idx]
        user_idx = self.user_id2user_idx[review_json["user_id"]]
        business_idx = self.business_id2business_idx[review_json["business_id"]]
        rating = review_json["stars"]
        return (user_idx, business_idx, rating)

    def __len__(self):
        return len(self.review_json_list)

    def get_num_user(self):
        return len(self.user_ids)

    def get_num_item(self):
        return len(self.business_ids)
