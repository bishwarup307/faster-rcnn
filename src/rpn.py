from __future__ import absolute_import, division, print_function
import torch
import numpy as np
from torchvision.models import resnet34
import torch.nn as nn
from typing import List, Union, Sequence, TypeVar, Any, Optional, Tuple

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
#ctypedef np

@cython.boundscheck(False)
@cython.wraparound(False)
def bbox_overlaps(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def xy_to_hw(anchor: Sequence[int]) -> Sequence[int]:
    """given a anchor in the `(xmin, ymin, xmax, ymax)` format transforms it to
        `(x_center, y_center, width, height)` format
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return x_ctr, y_ctr, w, h


def generate_anchors(
    stride: int = 16,
    scales: List[int] = [8, 16, 32],
    ratios: List[int] = [1, 0.5, 2],
    base_dist: Optional[int] = None,
) -> np.ndarray:
    """computes anchor boxes for a given stride, scales and aspect ratios.
        Number of anchors = stride ** 2 * scales * aspect_ratio

    Keyword Arguments:
        stride {int} -- the subsampling ratio of the output feature map (default: {16})
        scales {List[int]} -- list of scales (default: {[8, 16, 32]})
        ratios {List[int]} -- list of aspect ratios (default: {[1, 0.5, 2]})

    Returns:
        [np.ndarray] -- shape: `(stride * stride * scales * ratios, 4)`
    """
    base_dist = base_dist if base_dist is not None else stride

    x_out, y_out = stride, stride
    base_h, base_w = base_dist, base_dist
    y, x = np.meshgrid(np.arange(x_out), np.arange(y_out))

    x = x.reshape(-1,).repeat(len(scales) * len(ratios))
    y = y.reshape(-1,).repeat(len(scales) * len(ratios))
    x = x * base_w + base_w / 2
    y = y * base_h + base_h / 2

    ratios = np.tile(np.array(ratios), x_out * y_out).repeat(3)
    scales = np.tile(np.array(scales), x_out * y_out * 3)

    hs = np.round(base_h * scales * ratios).astype(int)
    ws = np.round(base_w * scales / ratios).astype(int)

    return np.vstack([x, y, ws, hs]).T


def hw_to_minmax(bbox: np.ndarray, max_dim: Tuple[int, int]) -> np.ndarray:
    """[summary]
    Args:
        bbox (np.ndarray): [description]
        max_dim (Tuple[int, int]): [description]

    Returns:
        np.ndarray: [description]
    """
    boxes = np.zeros_like(bbox, dtype=np.int32)
    max_x, max_y = max_dim

    boxes[:, 0], boxes[:, 2] = (
        np.maximum(bbox[:, 0] - bbox[:, 2] / 2, 0).astype(int),
        np.minimum(bbox[:, 0] + bbox[:, 2] / 2, max_x).astype(int),
    )

    boxes[:, 1], boxes[:, 3] = (
        np.maximum(bbox[:, 1] - bbox[:, 3] / 2, 0).astype(int),
        np.minimum(bbox[:, 1] + bbox[:, 3] / 2, max_y).astype(int),
    )
    return boxes


class RPN(nn.Module):
    def __init__(self, **kwargs):
        # self.input_size = input_size
        super(RPN, self).__init__()
        self.__dict__.update(kwargs)
        self.base = resnet34(pretrained=False)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


if __name__ == "__main__":
    # inp = torch.Tensor(1, 3, 1024, 1024)
    # model = RPN()  # resnet34(pretrained=False)
    # print(model(inp).size())
    # base_anchor = (0, 0, 15, 15)
    # print(xy_to_hw(base_anchor))
    anchors = generate_anchors(stride=32)
    print(anchors[:10])
    trsf_anchors = hw_to_minmax(anchors, (1024, 1024))
    print(trsf_anchors[:10])
