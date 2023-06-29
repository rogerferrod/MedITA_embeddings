import torch
import torch.nn as nn

class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, epsilon=0.1, thresh=0.5, writer=None):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = thresh
        self.margin = epsilon

        self.scale_pos = alpha
        self.scale_neg = beta
        
        # STATISTICS
        self.s = 0
        self.writer=writer

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        
        # Feature normalize
        x_norm = torch.norm(feats, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(feats, x_norm)

        sim_mat = torch.matmul(x_norm, torch.t(x_norm))

        epsilon = 1e-5
        loss = []
        
        # STATISTICS
        avg_negatives_mined = 0
        avg_positives_mined = 0
        avg_max_negative = 0
        avg_min_positive = 0
        # ---------

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]
                        
            if len(neg_pair_) >= 1:
                max_negative_sim = max(neg_pair_)
                pos_pair = pos_pair_[pos_pair_ - self.margin < max_negative_sim]
                if len(pos_pair) >= 1:
                    pos_loss = 1.0 / self.scale_pos * torch.log(
                        1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
                    loss.append(pos_loss)
                # STATISTICS
                avg_positives_mined += len(pos_pair)
                avg_max_negative += max_negative_sim
                # ---------
            if len(pos_pair_) >= 1:
                min_positive_sim = min(pos_pair_)
                neg_pair = neg_pair_[neg_pair_ + self.margin > min_positive_sim]
                if len(neg_pair) >= 1:
                    neg_loss = 1.0 / self.scale_neg * torch.log(
                        1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
                    loss.append(neg_loss)
                # STATISTICS
                avg_negatives_mined += len(neg_pair)
                avg_min_positive += min_positive_sim
                # ---------
        # STATISTICS
        self.writer.add_scalar('avg_negatives_mined', avg_negatives_mined/batch_size, global_step=self.s)
        self.writer.add_scalar('avg_positives_mined', avg_positives_mined/batch_size, global_step=self.s)
        self.writer.add_scalar('avg_max_negative', avg_max_negative/batch_size, global_step=self.s)
        self.writer.add_scalar('avg_min_positive', avg_min_positive/batch_size, global_step=self.s)
        self.s+=1
        # ---------
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).to(feats.device)

        loss = sum(loss) / batch_size
        return loss
    
class MultiSimilarityLossV2(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, epsilon=0.1, thresh_p=0.5, thresh_n=0.5, writer=None):
        super(MultiSimilarityLossV2, self).__init__()
        self.thresh_p = thresh_p
        self.thresh_n = thresh_n
        
        self.margin = epsilon

        self.scale_pos = alpha
        self.scale_neg = beta
        
        # STATISTICS
        self.s = 0
        self.writer=writer

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        
        # Feature normalize
        x_norm = torch.norm(feats, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(feats, x_norm)

        sim_mat = torch.matmul(x_norm, torch.t(x_norm))

        epsilon = 1e-5
        loss = []
        
        # STATISTICS
        avg_negatives_mined = 0
        avg_positives_mined = 0
        avg_max_negative = 0
        avg_min_positive = 0
        # ---------

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]
                        
            if len(neg_pair_) >= 1:
                max_negative_sim = max(neg_pair_)
                pos_pair = pos_pair_[pos_pair_ - self.margin < max_negative_sim]
                if len(pos_pair) >= 1:
                    pos_loss = 1.0 / self.scale_pos * torch.log(
                        1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh_p))))
                    loss.append(pos_loss)
                # STATISTICS
                avg_positives_mined += len(pos_pair)
                avg_max_negative += max_negative_sim
                # ---------
            if len(pos_pair_) >= 1:
                min_positive_sim = min(pos_pair_)
                neg_pair = neg_pair_[neg_pair_ + self.margin > min_positive_sim]
                if len(neg_pair) >= 1:
                    neg_loss = 1.0 / self.scale_neg * torch.log(
                        1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh_n))))
                    loss.append(neg_loss)
                # STATISTICS
                avg_negatives_mined += len(neg_pair)
                avg_min_positive += min_positive_sim
                # ---------
        # STATISTICS
        self.writer.add_scalar('avg_negatives_mined', avg_negatives_mined/batch_size, global_step=self.s)
        self.writer.add_scalar('avg_positives_mined', avg_positives_mined/batch_size, global_step=self.s)
        self.writer.add_scalar('avg_max_negative', avg_max_negative/batch_size, global_step=self.s)
        self.writer.add_scalar('avg_min_positive', avg_min_positive/batch_size, global_step=self.s)
        self.s+=1
        # ---------
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).to(feats.device)

        loss = sum(loss) / batch_size
        return loss
