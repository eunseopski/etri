import pdb
from math import cos, sin

import numpy as np
import scipy.io as sio
import cv2
import random
import torch

def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 0, 255), 3)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x1 - face_x), int(y2 + y1 - face_y)), (0, 0, 255), 3)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x2 - face_x), int(y1 + y2 - face_y)), (0, 0, 255), 3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), 2)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x3 - face_x), int(y1 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x3 - face_x), int(y2 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (255, 0, 0), 2)
    # Draw top in green
    cv2.line(img, (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x1 - face_x), int(y3 + y1 - face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x2 - face_x), int(y3 + y2 - face_y)), (0, 255, 0), 2)

    return img


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 4)

    return img


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params


def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params


def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d


# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def project_to_rotation_matrix_torch(R):
    """
    torch 버전 (구버전 호환): 주어진 (B,3,3) 행렬을 가장 가까운 rotation matrix로 보정
    """
    B = R.shape[0]
    R_proj = torch.zeros_like(R)
    for i in range(B):
        u, s, v = torch.svd(R[i])
        r = u @ v.t()
        if torch.det(r) < 0:
            u[:, -1] *= -1
            r = u @ v.t()
        R_proj[i] = r
    return R_proj


def wrap_angle(angle, limit=180):
    """
    angle: degree
    limit: 최대 절대값 (예: 180)
    """
    return ((angle + limit) % (2 * limit)) - limit


def compute_euler_angles_from_rotation_matrices(R_batch):
    """
    R_batch: (B,3,3) rotation matrices, 머리 local -> camera
    return: (B,3) [pitch, yaw, roll] in degrees
    """
    B = R_batch.shape[0]
    R_batch = project_to_rotation_matrix_torch(R_batch)

    out_euler = torch.zeros((B, 3), device=R_batch.device)

    for i in range(B):
        R = R_batch[i]
        if torch.abs(R[0, 2]) > 0.9999999:
            # gimbal lock
            z = 0.0
            if R[0, 2] > 0:
                y = -np.pi / 2
                x = torch.atan2(-R[1, 0], -R[2, 0])
            else:
                y = np.pi / 2
                x = torch.atan2(R[1, 0], R[2, 0])
            x_deg = np.rad2deg(x.cpu().numpy())
            y_deg = -np.rad2deg(y)
            out_euler[i, 0] = x_deg
            out_euler[i, 1] = y_deg
            out_euler[i, 2] = -0.0  # 그대로
        else:
            y0 = torch.asin(-R[0, 2])
            y1 = np.pi - y0
            cy0 = torch.cos(y0)
            cy1 = torch.cos(y1)

            x0 = torch.atan2(R[1, 2] / cy0, R[2, 2] / cy0)
            x1 = torch.atan2(R[1, 2] / cy1, R[2, 2] / cy1)

            z0 = torch.atan2(R[0, 1] / cy0, R[0, 0] / cy0)
            z1 = torch.atan2(R[0, 1] / cy1, R[0, 0] / cy1)

            x0_deg = np.rad2deg(x0.cpu().numpy())
            y0_deg = -np.rad2deg(y0.cpu().numpy())
            z0_deg = -np.rad2deg(z0.cpu().numpy())

            x1_deg = np.rad2deg(x1.cpu().numpy())
            y1_deg = -np.rad2deg(y1.cpu().numpy())
            z1_deg = -np.rad2deg(z1.cpu().numpy())

            if abs(z0_deg) < 90 and abs(x0_deg) < 90:
                out_euler[i, 0] = float(x0_deg)
                out_euler[i, 1] = float(y0_deg)
                out_euler[i, 2] = float(z0_deg)
            elif abs(z1_deg) < 90 and abs(x1_deg) < 90:
                out_euler[i, 0] = float(x1_deg)
                out_euler[i, 1] = float(y1_deg)
                out_euler[i, 2] = float(z1_deg)
            else:
                # out_euler[i, :] = torch.tensor([float('nan'), float('nan'), float('nan')], device=R_batch.device)
                out_euler[i, 0] = float(x0_deg)
                out_euler[i, 1] = float(y0_deg)
                out_euler[i, 2] = float(z0_deg)

    out_euler[:, 0] = wrap_angle(out_euler[:, 0])  # pitch
    out_euler[:, 1] = wrap_angle(out_euler[:, 1])  # yaw
    out_euler[:, 2] = wrap_angle(out_euler[:, 2])  # roll

    return out_euler  # degree


def compute_rotation_matrix_from_euler_angles(euler_batch, normalized=False):
    """
    euler_batch: (B,3) or (B,3,1)
      - pitch(X축), yaw(Y축), roll(Z축)
    normalized=True 면 0~1 입력으로 가정하고 denormalize 후 degree 변환
    return: (B,3,3) rotation matrices
    """

    if normalized:
        pitch_deg = (euler_batch[:, 0] - 0.5) * 180
        yaw_deg   = (euler_batch[:, 1] - 0.5) * 360
        roll_deg  = (euler_batch[:, 2] - 0.5) * 180
    else:
        pitch_deg = euler_batch[:, 0]
        yaw_deg   = euler_batch[:, 1]
        roll_deg  = euler_batch[:, 2]

    # degree → radian
    pitch = pitch_deg * (np.pi / 180.0)
    yaw   = -yaw_deg * (np.pi / 180.0)
    roll  = -roll_deg * (np.pi / 180.0)

    B = euler_batch.shape[0]
    rot_mats = torch.zeros((B, 3, 3), device=euler_batch.device, dtype=euler_batch.dtype)

    for i in range(B):
        cx = torch.cos(pitch[i])
        sx = torch.sin(pitch[i])
        cy = torch.cos(yaw[i])
        sy = torch.sin(yaw[i])
        cz = torch.cos(roll[i])
        sz = torch.sin(roll[i])

        # ZYX
        Rz = torch.tensor([
            [cz, -sz, 0],
            [sz,  cz, 0],
            [0,    0, 1]
        ], device=euler_batch.device, dtype=euler_batch.dtype)

        Ry = torch.tensor([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ], device=euler_batch.device, dtype=euler_batch.dtype)

        Rx = torch.tensor([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ], device=euler_batch.device, dtype=euler_batch.dtype)

        rot_mats[i] = Rz @ Ry @ Rx
        rot_mats = rot_mats.transpose(1, 2)

    euler_degree = torch.stack([pitch_deg, yaw_deg, roll_deg], dim=1)
    return rot_mats, euler_degree


def get_R(x, y, z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R


def draw_axis_orthographic_projection(img, rotation, img_center, thickness=3, axis_length=30, use_gray=False):
    '''
    img: image
    T_cam_head: [4,4], np.ndarray, this matrix transform head space points into camera space points
                [R t]
                [0 1]
    K: [3,4], intrinsic
        [f 0 cx 0]
        [0 f cy 0]
        [0 0 1 0]
    '''
    ####################################################################################
    # head space axis
    ####################################################################################
    origin_axis = np.array([0.0, 0.0, 0.0])
    x_axis = np.array([axis_length, 0.0, 0.0])
    y_axis = np.array([0.0, axis_length, 0.0])
    z_axis = np.array([0.0, 0.0, axis_length])
    axis_head = np.stack([origin_axis, x_axis, y_axis, z_axis], axis=0)  # [4,3]


    ####################################################################################
    # Transform axis
    ####################################################################################
    axis_cam = axis_head @ rotation.T
    axis_img = axis_cam[:,:2]

    img_center_ = np.array(img_center).reshape(1,2)
    axis_img = axis_img + img_center_

    # axis_img = axis_cam @ K.T
    # axis_img[:, :2] = axis_img[:, :2] / axis_img[:, [2]]
    # axis_img = axis_img[:, :2]
    #
    origin_img, x_axis_img, y_axis_img, z_axis_img = axis_img.astype(np.int64)

    ####################################################################################
    # Draw axis
    ####################################################################################
    if use_gray:
        color_gray = (127, 127, 127)
        color_x = color_gray
        color_y = color_gray
        color_z = color_gray
    else:
        color_x = (0, 0, 255)  # r
        color_y = (0, 255, 0)  # g
        color_z = (255, 0, 0)  # b

    axis_on_img = img.copy()
    axis_on_img = cv2.line(axis_on_img, origin_img, y_axis_img, color=color_y, thickness=thickness)  # g
    axis_on_img = cv2.line(axis_on_img, origin_img, z_axis_img, color=color_z, thickness=thickness)  # b
    axis_on_img = cv2.line(axis_on_img, origin_img, x_axis_img, color=color_x, thickness=thickness)  # r

    return axis_on_img


def denormalize(tensor, mean, std):
    """
    tensor: (B,3,H,W)
    mean: list of means
    std: list of stds
    """
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean


def plot_3axis_Zaxis(img, pitch, yaw, roll, tdx=None, tdy=None, size=50., limited=True, thickness=2):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]
    from math import cos, sin
    import math

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180

    if tdx != None and tdy != None:
        face_x = tdx
        face_y = tdy
    else:
        height, width = img.shape[:2]
        face_x = width / 2
        face_y = height / 2
    if tdx is None:
        tdx = face_x
    if tdy is None:
        tdy = face_y
    # X-Axis (pointing to right) drawn in red
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y

    # Y-Axis (pointing to down) drawn in green
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Plot head oritation line in black
    # scale_ratio = 5
    scale_ratio = 2
    base_len = math.sqrt((face_x - x3) ** 2 + (face_y - y3) ** 2)
    if face_x == x3:
        endx = tdx
        if face_y < y3:
            if limited:
                endy = tdy + (y3 - face_y) * scale_ratio
            else:
                endy = img.shape[0]
        else:
            if limited:
                endy = tdy - (face_y - y3) * scale_ratio
            else:
                endy = 0
    elif face_x > x3:
        if limited:
            endx = tdx - (face_x - x3) * scale_ratio
            endy = tdy - (face_y - y3) * scale_ratio
        else:
            endx = 0
            endy = tdy - (face_y - y3) / (face_x - x3) * tdx
    else:
        if limited:
            endx = tdx + (x3 - face_x) * scale_ratio
            endy = tdy + (y3 - face_y) * scale_ratio
        else:
            endx = img.shape[1]
            endy = tdy - (face_y - y3) / (face_x - x3) * (tdx - endx)
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,0,0), 2)
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (255,255,0), 2)
    # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0, 255, 255), thickness)

    # X-Axis pointing to right. drawn in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), thickness)
    # Y-Axis pointing to down. drawn in green
    cv2.line(img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 255, 0), thickness)
    # Z-Axis (out of the screen) drawn in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), thickness)

    return img


def vis(images, pred_mat, targets_deg, targets_rotation):
    # 이미지 부터 제대로 출력
    # 1. 모델이 예측한 rotation을 시각화 및 출력
    # 2. 모델이 예측한 rotation을 euler angle로 바꿔서 시각화 및 출력
    # 3. targets euler angle을 시각화 및 출력
    # 4. targets euler angle을 rotation matrix로 바꿔서 시각화 및 출력

    denorm_images = denormalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = denorm_images[0].detach().cpu().numpy()  # [3, H, W]
    img = (img * 255).astype(np.uint8)  # float → uint8
    img = np.transpose(img, (1, 2, 0))  # ➡ [H, W, 3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_center = img.shape[0] // 2, img.shape[1] // 2

    # 모델이 예측한 rotation 시각화
    pred_mat = pred_mat[:, :].detach().cpu().numpy()  # [3,3]
    img_pred_rotation = draw_axis_orthographic_projection(img.copy(), pred_mat, img_center, axis_length=45)

    # 모델이 예측한 rotation -> euler 시각화
    if isinstance(pred_mat, np.ndarray):
        pred_mat = torch.from_numpy(pred_mat)
    pred_mat = pred_mat.unsqueeze(0)
    pred_euler = compute_euler_angles_from_rotation_matrices(pred_mat).squeeze()
    img_pred_euler = plot_3axis_Zaxis(img.copy(), pred_euler[0], pred_euler[1], pred_euler[2], size=50., limited=True, thickness=2)

    # GT target euler angle 시각화
    img_target_euler = plot_3axis_Zaxis(img.copy(), targets_deg[0], targets_deg[1], targets_deg[2], size=50., limited=True, thickness=2)

    # GT target rotation 시각화
    if isinstance(targets_deg, np.ndarray):
        targets_deg = torch.from_numpy(targets_deg)
    targets = targets_deg.unsqueeze(0)
    rot_targets, _ = compute_rotation_matrix_from_euler_angles(targets, normalized=False)  # normalized euler angle -> (rotation matrix, radian auler angle)
    rot_targets = rot_targets.squeeze(0).detach().cpu().numpy()
    img_targets_rotation = draw_axis_orthographic_projection(img.copy(), rot_targets, img_center, axis_length=45)

    # targets_rotation = targets_rotation[:, :].detach().cpu().numpy()  # [3,3]
    # img_targets_rotation = draw_axis_orthographic_projection(img.copy(), targets_rotation, img_center, axis_length=45)

    # # GT target rotation -> euler angle로 변환 후 시각화
    # 모델이 예측한 rotation -> euler 시각화
    if isinstance(targets_rotation, np.ndarray):
        targets_rotation = torch.from_numpy(targets_rotation)
    targets_rotation = targets_rotation.unsqueeze(0)
    pred_rot2euler = compute_euler_angles_from_rotation_matrices(targets_rotation).squeeze()
    # img_pred_rot2euler = plot_3axis_Zaxis(img.copy(), pred_rot2euler[0], pred_rot2euler[1], pred_rot2euler[2], size=50., limited=True, thickness=2)


    print(f'Pred_euler: {pred_euler}')
    print(f'target_euler: {targets_deg}')
    print(f'Pred_rotation: {pred_mat}')
    print(f'target_rotation: {targets_rotation}')
    print(f'target_rot2euler: {pred_rot2euler}')


    cv2.imshow("Pred_euler", img_pred_euler)
    cv2.imshow("Pred_rotation", img_pred_rotation)
    cv2.imshow("target_euler", img_target_euler)
    cv2.imshow("target_rotation", img_targets_rotation)
    # cv2.imshow("target_rot2euler", img_pred_rot2euler)

    cv2.waitKey(0)


def compute_euler_angles_from_rotation_matrices_other(rotation_matrices, full_range=False, use_gpu=True, gpu_id=0):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    '''2023.01.15'''
    for i in range(len(sy)):  # expand y (yaw angle) range into (-180, 180)
        if R[i, 0, 0] < 0 and full_range:
            sy[i] = -sy[i]

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)  # sy > 0, so y (yaw angle) is always in range (-90, 90)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)  # sy > 0, so y (yaw angle) is always in range (-90, 90)
    zs = R[:, 1, 0] * 0

    if use_gpu:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3).cuda(gpu_id))
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3))
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular

    return out_euler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False