import torch
import torch.nn as nn


class ProbabilisticMatrixFactorization(nn.Module):
    def __init__(self, num_user, num_item, latent_dim):
        super(ProbabilisticMatrixFactorization, self).__init__()
        self.w_user = nn.parameter.Parameter(
            torch.empty((num_user, latent_dim)), requires_grad=True,
        )
        self.w_item = nn.parameter.Parameter(
            torch.empty((num_item, latent_dim)), requires_grad=True,
        )
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.w_user, mean=0.0, std=0.1)
        nn.init.normal_(self.w_item, mean=0.0, std=0.1)

    def forward(self, user_indices, item_indices):
        user_embeds = torch.index_select(self.w_user, 0, user_indices)
        item_embeds = torch.index_select(self.w_item, 0, item_indices)
        estimate_ratings = torch.sum(user_embeds * item_embeds, dim=-1)
        return estimate_ratings
