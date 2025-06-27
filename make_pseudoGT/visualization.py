import os
import cv2
import numpy as np
import glob

# 경로 설정
label_dir = '/home/choi/hwang/workspace/etri/make_pseudoGT/dataset/full_range_head/AGORA/validation/labels'
image_dir = '/home/choi/hwang/workspace/etri/make_pseudoGT/dataset/full_range_head/AGORA/validation/images'

# 시각화 저장할 경로
save_dir = './headpose_visualized_128x128'
os.makedirs(save_dir, exist_ok=True)

# degree to radian 변환 함수
def deg2rad(deg):
    return deg * np.pi / 180

# Head pose 시각화용 축 그리기
import cv2
import numpy as np


def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size=40, normalize=True):
    """
    정규화된 head pose (yaw, pitch, roll)를 기반으로 시각화 선을 그려주는 함수

    yaw   : 좌우 회전 (정규화된 값: 0~1, 실제각도 -180 ~ 180)
    pitch : 위아래 회전 (정규화된 값: 0~1, 실제각도 -90 ~ 90)
    roll  : 기울임 (정규화된 값: 0~1, 실제각도 -90 ~ 90)
    """
    if normalize:
        # 정규화 값 → 실제 각도로 변환
        pitch = (pitch - 0.5) * 180  # Y축 회전: 위아래
        yaw = (yaw - 0.5) * 360  # Z축 회전: 좌우
        roll = (roll - 0.5) * 180  # X축 회전: 기울임

    # 라디안 변환 (OpenCV 좌표계에 맞게 yaw는 부호 반전)
    pitch = np.radians(pitch)
    yaw = -np.radians(yaw)
    roll = np.radians(roll)

    if tdx is None or tdy is None:
        h, w = img.shape[:2]
        tdx = w / 2
        tdy = h / 2

    # 방향벡터 계산
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    # 시각화 선 그리기
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)  # X축 (빨강)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)  # Y축 (초록)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)  # Z축 (파랑)

    return img


# main 루프
txt_list = sorted(glob.glob(os.path.join(label_dir, '*.txt')))[:300]

for txt_path in txt_list:
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    img_path = os.path.join(image_dir, base_name + '.jpg')
    if not os.path.exists(img_path):
        img_path = os.path.join(image_dir, base_name + '.png')
        if not os.path.exists(img_path):
            print(f"이미지 없음: {base_name}")
            continue

    # 이미지 로드 및 리사이즈
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # 👈 여기!

    # 라벨 읽기
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
        pitch, yaw, roll = map(float, line.split())

    # 시각화 (이미지 중심 기준)
    img = draw_axis(img, pitch, yaw, roll, tdx=64, tdy=64)

    # 저장
    save_path = os.path.join(save_dir, base_name + '_pose.jpg')
    cv2.imwrite(save_path, img)

# print("✅ 128x128 리사이즈 후 시각화 완료!")