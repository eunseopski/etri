import os

# 원래 절대경로가 들어있는 파일
input_txt = "/home/choi/hwang/workspace/etri/make_pseudoGT/dataset/CMU-HPE/yolov5_labels_coco/img_txt/val.txt"

# 새로 만들 파일 (파일명만 있는 리스트)
output_txt = "/home/choi/hwang/workspace/etri/make_pseudoGT/dataset/CMU-HPE/yolov5_labels_coco/img_txt/relative_validation.txt"

# 처리
with open(input_txt, 'r') as f:
    lines = f.readlines()

# 파일 이름만 추출
filenames = [os.path.basename(line.strip()) for line in lines]

# 저장
with open(output_txt, 'w') as f:
    for name in filenames:
        f.write(name + '\n')

print(f"✅ 파일명만 저장 완료: {output_txt}")
