import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(target, predicted):
    y_pred = predicted.argmax(dim=1).flatten().to("cpu")
    y_true = target.argmax(dim=1).flatten().to("cpu")

    confusion_tensor = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    intersection = np.diag(confusion_tensor)
    ground_truth_set = confusion_tensor.sum(axis=1)

    # Compute Mean Precision Accuracy
    mpa = (intersection / ground_truth_set.astype(np.float32)).mean()

    predicted_set = confusion_tensor.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    # Compute the IoU
    iou = intersection / union.astype(np.float32)
    m_iou = iou.mean()

    # Compute Frequency Weighted IoU
    total_pixels = ground_truth_set.sum()
    fw_iou = np.dot(ground_truth_set.astype(float) / total_pixels, iou)

    return mpa, m_iou, fw_iou
