import torch
import torch.nn as nn

class IoU(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoU, self).__init__()
        self.threshold = threshold

    def forward(self, target, input):
        eps = 1e-10
        input_ = (input > self.threshold).data.float()
        target_ = (target > self.threshold).data.float()

        intersection = torch.clamp(input_ * target_, 0, 1) # xi < 0, xi = 0; xi > 1, xi = 1; 0 < xi < 1, xi = xi
        union = torch.clamp(input_ + target_, 0, 1)

        if torch.mean(intersection).lt(eps):
            return torch.Tensor([0., 0., 0., 0., 0., 0.])
        else:
            acc = torch.mean((input_ == target_).data.float())
            if len(target.shape) == 2:
                iou = intersection.sum() / union.sum()
                inter_all = intersection.sum()
                union_all = union.sum()
            else:
                iou = torch.sum(intersection.sum(2).sum(2) / union.sum(2).sum(2))
                inter_all = torch.sum(intersection.sum(2).sum(2))
                union_all = torch.sum(union.sum(2).sum(2))
            recall = torch.mean(intersection) / torch.mean(target_)
            precision = torch.mean(intersection) / torch.mean(input_)
            return torch.Tensor([acc, recall, precision, iou, inter_all, union_all])
