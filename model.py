import torch
import torch.nn as nn


class ProbabilisticMatrixFactorization(nn.Module):
    def __init__(self, num_user, num_item, hidden_dim):
        super(ProbabilisticMatrixFactorization, self).__init__()
        self.w_user = nn.parameter.Parameter(
            torch.empty((num_user, hidden_dim), device=torch.device("cpu"))
        )
        self.w_item = nn.parameter.Parameter(
            torch.empty((num_item, hidden_dim), device=torch.device("cpu"))
        )
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.w_user)
        nn.init.normal_(self.w_item)

    def forward(self, user_indices, item_indices):
        user_embeds = self.w_user[user_indices]
        item_embeds = self.w_item[item_indices]
        estimate_ratings = user_embeds * item_embeds
        return estimate_ratings
