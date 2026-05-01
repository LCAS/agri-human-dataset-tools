"""Minimal MMDetection3D dataset class for the exported Agri-Human person dataset.

Copy this file into your MMDetection3D checkout, for example:
  mmdetection3d/mmdet3d/datasets/agri_person_dataset.py
and import/register it from that environment.
"""

from __future__ import annotations

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes

from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class AgriPersonDataset(Det3DDataset):
    METAINFO = {
        "classes": ("person",),
    }

    def parse_ann_info(self, info):
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            ann_info["gt_bboxes_3d"] = np.zeros((0, 7), dtype=np.float32)
            ann_info["gt_labels_3d"] = np.zeros(0, dtype=np.int64)
        ann_info = self._remove_dontcare(ann_info)
        ann_info["gt_bboxes_3d"] = LiDARInstance3DBoxes(ann_info["gt_bboxes_3d"])
        return ann_info
