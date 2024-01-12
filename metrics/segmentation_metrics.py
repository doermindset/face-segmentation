import numpy as np


def mean_pixel_accuracy(gt, pred):
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    predicted_classes = np.argmax(pred_np, axis=1)
    gt_classes = np.argmax(gt_np, axis=1)

    correct_pixels = np.sum(predicted_classes == gt_classes)
    total_pixels = gt_np.size // gt_np.shape[1]

    mean_pixel_acc = correct_pixels / total_pixels

    return mean_pixel_acc


def mean_iou(segs, segs_pred):
    segs_np = segs.clone().detach().cpu().numpy().argmax(1)
    segs_pred_np = segs_pred.clone().detach().cpu().numpy().argmax(1)

    intersection = np.logical_and(segs_np, segs_pred_np).sum((1, 2))
    union = np.logical_or(segs_np, segs_pred_np).sum((1, 2))

    iou_score = np.where(union == 0, 0, intersection / union)
    m_iou = iou_score.mean()

    return m_iou

def frequency_weighted_iou(segs, segs_pred):
    intersection = np.logical_and(segs, segs_pred)
    union = np.logical_or(segs, segs_pred)

    class_intersection = np.sum(intersection, axis=(1, 2))
    class_union = np.sum(union, axis=(1, 2))
    class_iou = class_intersection / class_union

    class_frequency = np.sum(segs, axis=(1, 2)) / np.prod(segs.shape[1:])

    weighted_iou = class_iou * class_frequency
    fw_iou = np.sum(weighted_iou) / np.sum(class_frequency)

    return fw_iou


def compute_metrics(segs, segs_pred):

    mpa = mean_pixel_accuracy(segs, segs_pred)
    m_iou = mean_iou(segs, segs_pred)

    return mpa, m_iou
