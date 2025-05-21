import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF

def visualize_sample(dataset, index=0):
    img, target = dataset[index]  # __getitem__ 실행
    boxes = target['boxes']       # torch.Tensor 형태 (N, 4)
    labels = target['labels']     # torch.Tensor 형태 (N,)

    # img: Tensor(C, H, W) -> numpy(H, W, C)
    img_np = TF.to_pil_image(img).convert("RGB")

    # 시각화
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_np)

    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.set_title(f"Augmented Sample #{index}")
    plt.axis('off')
    plt.show()