import torch.nn as nn
from model.utils import compute_rotation_matrix_from_euler
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
    return torch.mean(wrapped_diff)  # 전체 평균 (pitch, yaw, roll 포함)


def compute_maev(pred_rot, gt_rot, eps=1e-6):
    """
    MAEV: Mean Absolute Error of Vectors (3-axis 평균 버전)
    회전행렬의 3개 축 벡터를 각각 비교해 평균 오차를 구함.

    Args:
        pred_rot (Tensor): [B, 3, 3], 예측 회전 행렬
        gt_rot   (Tensor): [B, 3, 3], GT 회전 행렬
        eps (float): 안정성 위한 epsilon

    Returns:
        maev (Tensor): 평균 각도 차이 (degree)
    """
    axis_errors = []

    for axis in range(3):
        pred_v = pred_rot[:, :, axis]  # [B, 3]
        gt_v   = gt_rot[:, :, axis]

        # 정규화
        pred_v = F.normalize(pred_v, dim=1)
        gt_v   = F.normalize(gt_v, dim=1)

        # 벡터 사이 각도
        cos_sim = torch.sum(pred_v * gt_v, dim=1)  # [B]
        cos_sim = torch.clamp(cos_sim, -1 + eps, 1 - eps)

        angle = torch.acos(cos_sim)  # [B], radian
        axis_errors.append(angle)

    # 3축의 평균
    axis_errors = torch.stack(axis_errors, dim=1)  # [B, 3]
    mean_error_per_sample = torch.mean(axis_errors, dim=1)  # [B]
    maev_radian = torch.mean(mean_error_per_sample)  # scalar

    maev_degree = torch.rad2deg(maev_radian)
    return maev_degree
