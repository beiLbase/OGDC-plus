import sys
import torch
from torch import autograd, optim, nn
import numpy as np
import fewshot_re_kit
from torch.distributions import MultivariateNormal


class Proto(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, dot=False, k=50, alpha_dc=0.21, num_pseudo_samples=50):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.dot = dot
        self.hidden_size = 768
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.k = k
        self.alpha_dc = alpha_dc
        self.num_pseudo_samples = num_pseudo_samples

    def __dist__(self, x, y, dim):

        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):

        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def __euclid_dist__(self, x, y, dim):

        return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_euclid_dist__(self, S, Q):

        return self.__euclid_dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def distribution_calibration(self, support, query, k, alpha=None):

        B, N, K, D = support.size()
        _, total_Q, _ = query.size()


        support_ = support.view(B * N * K, D)
        query_ = query.view(B * total_Q, D)

        dist_matrix = torch.cdist(support_, query_)  # (B*N*K, B*total_Q)

        effective_k = min(k, dist_matrix.size(1))
        topk_dist, indices = torch.topk(dist_matrix, k=effective_k, dim=1, largest=False)

        nearest_query_samples = torch.gather(
            query_.unsqueeze(0).expand(B * N * K, -1, -1),
            1,
            indices.unsqueeze(-1).expand(-1, -1, D)
        )  # (B*N*K, k, D)


        exp_neg_dist = torch.exp(-topk_dist)
        sum_exp_neg_dist = torch.sum(exp_neg_dist, dim=1, keepdim=True) + 1e-8 
        weights = exp_neg_dist / sum_exp_neg_dist


        weighted_query_sum = torch.bmm(
            weights.unsqueeze(1),
            nearest_query_samples
        ).squeeze(1)  # (B*N*K, D)

        calibrated_means = 0.5 * (support_ + weighted_query_sum)


        query_vars = torch.var(nearest_query_samples, dim=1, unbiased=False)
        global_vars = torch.var(support_, dim=0, unbiased=False)
        calibrated_vars = global_vars + (alpha if alpha is not None else 0.0) * query_vars
        calibrated_covs = torch.stack([torch.diag(v) for v in calibrated_vars], dim=0)

    
        calibrated_means = calibrated_means.view(B, N, K, D)
        calibrated_covs = calibrated_covs.view(B, N, K, D, D)

        return calibrated_means, calibrated_covs

    def generate_pseudo_samples(self, means, covs, num_samples):

        B, N, K, D = means.size()
        pseudo_samples = []
        for b in range(B):
            for n in range(N):
                for k_idx in range(K):
                    mean = means[b, n, k_idx]
                    cov = covs[b, n, k_idx]

                    dist = MultivariateNormal(mean, covariance_matrix=cov)
                    samples = dist.sample((num_samples,))
                    pseudo_samples.append(samples)


        pseudo_samples = torch.stack(pseudo_samples, dim=0).view(B, N, -1, D)
        return pseudo_samples

    def forward(self, support, query, rel_txt, N, K, total_Q):

        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)
        rel_loc = torch.mean(rel_loc, 1)  # [B*N, D]
        rel_rep = torch.cat((rel_gol, rel_loc), -1)  # [B*N, 2D]

        support_h, support_t, s_loc = self.sentence_encoder(support)  # (B*N*K, D)
        query_h, query_t, q_loc = self.sentence_encoder(query)  # (B*total_Q, D)
        support = torch.cat((support_h, support_t), -1)  # 拼接头尾特征
        query = torch.cat((query_h, query_t), -1)
        support = support.view(-1, N, K, self.hidden_size * 2)  # (B, N, K, 2D)
        query = query.view(-1, total_Q, self.hidden_size * 2)  # (B, total_Q, 2D)


        support_ = support.view(-1, N * K, self.hidden_size * 2)
        dist_sq = self.__batch_euclid_dist__(support_, query).view(-1, total_Q, N, K)
        query_guided_weights = dist_sq.mean(1).tanh().softmax(-1).unsqueeze(-1)
        proto1_support = (support * query_guided_weights).sum(2)  # (B, N, 2D)

        rel_rep = rel_rep.view(-1, N, self.hidden_size * 2)
        alpha_sigmoid = torch.sigmoid(self.alpha)
        fused_support = alpha_sigmoid * proto1_support + (1 - alpha_sigmoid) * rel_rep

        fused_support = fused_support.unsqueeze(2)  # (B, N, 1, 2D)


        calibrated_means, calibrated_covs = self.distribution_calibration(
            fused_support, query, self.k, self.alpha_dc
        )

        pseudo_samples = self.generate_pseudo_samples(calibrated_means, calibrated_covs, self.num_pseudo_samples)
        combined_support = torch.cat([fused_support, pseudo_samples], dim=2)  # (B, N, 1+num_pseudo_samples, 2D)
        final_prototypes = torch.mean(combined_support, dim=2)  # (B, N, 2D)
        logits = self.__batch_dist__(final_prototypes, query)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N+1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        return logits, pred
