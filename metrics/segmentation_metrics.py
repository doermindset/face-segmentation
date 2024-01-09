import numpy as np

def mean_pixel_accuracy(segs, segs_pred):
    correct_pixels = np.sum((segs == segs_pred).astype(np.float32))
    total_pixels = np.prod(segs.shape)

    mean_accuracy = correct_pixels / total_pixels

    return mean_accuracy


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

    # segs_pred = F.softmax(segs_pred, dim=1)
    # segs = segs.clone().detach().cpu().numpy().argmax(1)
    # segs_pred = segs_pred.clone().detach().cpu().numpy().argmax(1)
    #
    #
    # segs_pred = segs_pred.contiguous().view(-1)
    # segs = segs.contiguous().view(-1)
    # segs_np = segs.cpu().detach().numpy()
    # segs_pred_np = segs_pred.cpu().detach().numpy()

    # assert segs_np.shape == segs_pred_np.shape, "Shapes of segs_np and segs_pred_np should be the same."

    # mpa = mean_pixel_accuracy(segs_np, segs_pred_np)
    # fw_iou = frequency_weighted_iou(segs_np, segs_pred_np)

    m_iou = mean_iou(segs, segs_pred)

    return m_iou
