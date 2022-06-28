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

    def forward(self, uesr_indices, item_indices):
        user_embeds = torch.index_select(self.w_user, 0, uesr_indices)
        item_embeds = torch.index_select(self.w_item, 0, item_indices)
        estimate_ratings = torch.sum(user_embeds * item_embeds, dim=-1)
        return estimate_ratings
