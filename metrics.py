from brambox.stat import coordinates
from brambox.stat._matchboxes import match_det, match_anno
import numpy as np

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def get_moda(det, anno, threshold=0.2, ignore=None):
    if ignore is None:
        ignore = anno.ignore.any()

    # dets_per_frame = anno.groupby('image').filter(lambda x: any(x['ignore'] == 0)) 안쓰이길래 주석처리
    # dets_per_frame = dets_per_frame.groupby('image').size().to_dict()

    # Other param for finding matched anno
    crit = coordinates.pdollar if ignore else coordinates.iou
    label = len({*det.class_label.unique(), *anno.class_label.unique()}) > 1
    matched_dets = match_det(det, anno, threshold, criteria=crit,
                            class_label=label, ignore=2 if ignore else 0)
    fp_per_im = matched_dets[matched_dets.fp==True].groupby('image').size().to_dict()
    tp_per_im = matched_dets[matched_dets.tp==True].groupby('image').size().to_dict()
    valid_anno = anno[anno.ignore == False].groupby('image').size().to_dict()

    # assert valid_anno.keys() == tp_per_im.keys()
    # 위 코드 대신 이미지 수 체크 정도로 바꾸는 게 좋음
    missing_tp_images = set(valid_anno.keys()) - set(tp_per_im.keys())
    if missing_tp_images:
        print(f"len(missing_tp_images): {len(missing_tp_images)}")

        print("missing_tp_images top10:")
        for img_id in sorted(missing_tp_images)[:10]:
            print(img_id)

    moda_ = []
    for k, _ in valid_anno.items():
        n_gt = valid_anno[k]
        tp = tp_per_im.get(k, 0)
        fp = fp_per_im.get(k, 0)
        # miss = n_gt-tp_per_im[k]
        miss = n_gt - tp
        # fp = fp_per_im[k]
        moda_.append(safe_div((miss+fp), n_gt))
    return 1 - np.mean(moda_)


def get_modp(det, anno, threshold=0.2, ignore=None):
    if ignore is None:
        ignore = anno.ignore.any()
    # Compute TP/FP
    if not {'tp', 'fp'}.issubset(det.columns):
        crit = coordinates.pdollar if ignore else coordinates.iou
        label = len({*det.class_label.unique(), *anno.class_label.unique()}) > 1
        det = match_anno(det, anno, threshold, criteria=crit, class_label=label, ignore=2 if ignore else 0)
    elif not det.confidence.is_monotonic_decreasing:
        det = det.sort_values('confidence', ascending=False)
    modp = det.groupby('image')['criteria'].mean().mean()
    return modp