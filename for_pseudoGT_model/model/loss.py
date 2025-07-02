import torch.nn as nn
import torch
import torch.nn.functional as F


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):

        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))

        return torch.mean(theta)

def compute_mawe(pred_deg, target_deg):
    """
    pred_deg: [B, 3] 예측값 (degree)
    target_deg: [B, 3] GT (degree)
    """
    diff = torch.abs(pred_deg - target_deg)
    wrapped_diff = torch.minimum(diff, 360.0 - diff)
    total_mae = torch.mean(wrapped_diff)  # 전체 평균
    return total_mae


def compute_maev(pred_rot, gt_rot, eps=1e-6):
    """
    MAEV: Mean Angular Error of Vectors (degree)
    회전행렬의 3축 벡터를 평균 각도 오차로 계산.

    Args:
        pred_rot (Tensor): [B, 3, 3]
        gt_rot   (Tensor): [B, 3, 3]
        eps (float): 안정성용 epsilon

    Returns:
        maev (Tensor): scalar (degree)
    """
    axis_errors = []

    for axis in range(3):
        pred_v = pred_rot[:, :, axis]
        gt_v   = gt_rot[:, :, axis]

        pred_v = F.normalize(pred_v, dim=1, eps=eps)
        gt_v   = F.normalize(gt_v, dim=1, eps=eps)

        cos_sim = torch.sum(pred_v * gt_v, dim=1).clamp(-1.0, 1.0)
        angle = torch.acos(cos_sim)  # radian
        axis_errors.append(angle)

    axis_errors = torch.stack(axis_errors, dim=1)  # [B, 3]
    mean_error_per_sample = axis_errors.mean(dim=1)  # [B]
    maev_degree = torch.rad2deg(mean_error_per_sample).mean()  # scalar

    return maev_degree

def compute_symmetric_maev(pred_rot, gt_rot, eps=1e-6):
    """
    Symmetry-aware MAEV
    pred_rot, gt_rot: [B, 3, 3]
    """
    axis_errors = []

    for axis in range(3):
        pred_v = pred_rot[:, :, axis]
        gt_v   = gt_rot[:, :, axis]

        pred_v = F.normalize(pred_v, dim=1, eps=eps)
        gt_v   = F.normalize(gt_v, dim=1, eps=eps)

        # dot product
        cos_sim = torch.sum(pred_v * gt_v, dim=1).clamp(-1.0, 1.0)
        # 축 뒤집힘까지 고려
        cos_sim_sym = torch.abs(cos_sim)

        angle = torch.acos(cos_sim_sym)
        axis_errors.append(angle)

    axis_errors = torch.stack(axis_errors, dim=1)
    mean_error_per_sample = axis_errors.mean(dim=1)
    maev_degree = torch.rad2deg(mean_error_per_sample).mean()
    return maev_degree
