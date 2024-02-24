import torch
import torch.nn as nn
import torch.nn.functional as F
import  numpy as np
import cv2
from abc import ABC
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, inputs, targets, weight):
        # BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        BCE_loss = nn.BCELoss(weight=weight)(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.size_average:
            return torch.mean(F_loss)
        else:
            return F_loss

class PixelContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PixelContrastLoss, self).__init__()

        self.temperature =  0.1
        self.base_temperature =  0.07

        self.ignore_label = -1

        self.max_samples = 1024
        self.max_views = 100

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

        self.contrast_criterion = PixelContrastLoss()


    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def generate_edge_tensor(self, label, edge_width=3):
        device = label.device
        label = label.type(torch.cuda.FloatTensor)
        if len(label.shape) == 2:
            label = label.unsqueeze(0)
        n, h, w = label.shape
        edge = torch.zeros(label.shape, dtype=torch.float).to(device) #.cuda()
        # right
        edge_right = edge[:, 1:h, :]
        edge_right[(label[:, 1:h, :] != label[:, :h - 1, :]) & (label[:, 1:h, :] != 255)
                   & (label[:, :h - 1, :] != 255)] = 1

        # up
        edge_up = edge[:, :, :w - 1]
        edge_up[(label[:, :, :w - 1] != label[:, :, 1:w])
                & (label[:, :, :w - 1] != 255)
                & (label[:, :, 1:w] != 255)] = 1

        # upright
        edge_upright = edge[:, :h - 1, :w - 1]
        edge_upright[(label[:, :h - 1, :w - 1] != label[:, 1:h, 1:w])
                     & (label[:, :h - 1, :w - 1] != 255)
                     & (label[:, 1:h, 1:w] != 255)] = 1

        # bottomright
        edge_bottomright = edge[:, :h - 1, 1:w]
        edge_bottomright[(label[:, :h - 1, 1:w] != label[:, 1:h, :w - 1])
                         & (label[:, :h - 1, 1:w] != 255)
                         & (label[:, 1:h, :w - 1] != 255)] = 1

        kernel = torch.ones((1, 1, edge_width, edge_width), dtype=torch.float).to(device) #.cuda()
        with torch.no_grad():
            edge = edge.unsqueeze(1)
            edge = F.conv2d(edge, kernel, stride=1, padding=1)
        edge[edge != 0] = 1
        edge = edge.squeeze()
        return edge

    def __call__(self, y_true, y_pred, epoch= 0):
        scale_pred, pred_edge, embedding = y_pred
        _, _, h, w = scale_pred.shape
        weight = torch.zeros_like(y_true).float().cuda()
        black_weight = (y_true == 1).sum() / ((y_true == 0).sum() + (y_true == 1).sum())
        road_weight = (y_true == 0).sum() / ((y_true == 0).sum() + (y_true == 1).sum())
        weight = torch.fill_(weight, black_weight)
        weight[y_true == 1.0] = road_weight
        a = nn.BCELoss(weight=weight)(scale_pred, y_true)
        b = self.soft_dice_loss(y_true, scale_pred)

        edge_true = self.generate_edge_tensor(y_true.squeeze(1))
        if len(edge_true.shape) == 2:
            edge_true = edge_true.unsqueeze(0).unsqueeze(0)
        else:
            edge_true = edge_true.unsqueeze(1)
        scale_pred_edge = F.interpolate(input=pred_edge, size=(h, w), mode='bilinear', align_corners=True)
        weight_edge = torch.zeros_like(edge_true).float().cuda()
        black_weight_edge = (edge_true == 1).sum() / ((edge_true == 0).sum() + (edge_true == 1).sum())
        road_weight_edge = (edge_true == 0).sum() / ((edge_true == 0).sum() + (edge_true == 1).sum())
        weight_edge = torch.fill_(weight_edge, black_weight_edge)
        weight_edge[edge_true == 1.0] = road_weight_edge
        edge_loss = nn.BCELoss(weight=weight_edge)(scale_pred_edge, edge_true)

        if embedding.shape[-1] != y_true.shape[-1]:
            _, _, emb_h, emb_w = embedding.shape
            y_true_emb = F.interpolate(input=y_true, size=(emb_h, emb_w), mode='bilinear', align_corners=True)
            scale_pred_emb = F.interpolate(input=scale_pred, size=(emb_h, emb_w), mode='bilinear', align_corners=True)
        else:
            scale_pred_emb = scale_pred
            y_true_emb = y_true
        predict = (scale_pred_emb > 0.5).data.float()

        img = np.asarray(y_true_emb.cpu())
        mask_list = []
        kernel = np.ones((8, 8), np.uint8)
        for idx in range(img.shape[0]):
            erosion = cv2.erode(img[idx][0], kernel, iterations=1)
            dilate = cv2.dilate(img[idx][0], kernel, iterations=1)
            mask = dilate - erosion
            mask[mask != 0] = -2
            mask_list.append(mask[np.newaxis, np.newaxis, ...])
        mask = np.concatenate(mask_list, axis=0)
        mask = torch.from_numpy(mask).to(y_true_emb.device)
        predict[mask != -2] = -1

        contrast_loss = self.contrast_criterion(embedding, y_true_emb, predict)

        if epoch > 100:
            return a + b + edge_loss * 0.5 + contrast_loss * 0.01
        else:
            return a + b + edge_loss * 0.5
