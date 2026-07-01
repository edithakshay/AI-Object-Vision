"""
YOLO26 Task Registry — modular interface for current and future task types.

Currently ACTIVE:
  ObjectDetection — bounding boxes + class + confidence

STUBBED (architecture ready, not yet enabled):
  InstanceSegmentation  — pixel masks per detection
  PoseEstimation        — keypoint skeletons
  ImageClassification   — whole-frame class label
  OrientedBoundingBox   — rotated bounding boxes (OBB)
  OpenVocabDetection    — YOLOE-26 open-vocabulary detection

To enable a task later:
  1. Set its TaskConfig.enabled = True in the registry.
  2. Implement run() in the task class.
  3. Export the correct model variant (e.g. yolo26n-seg.pt for segmentation).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger("DualVisionAI.tasks")


class TaskType(str, Enum):
    DETECT     = "detect"
    SEGMENT    = "segment"
    POSE       = "pose"
    CLASSIFY   = "classify"
    OBB        = "obb"
    OPEN_VOCAB = "open_vocab"


@dataclass
class TaskConfig:
    task_type:    TaskType
    enabled:      bool   = False
    model_suffix: str    = ""          # e.g. "-seg", "-pose", "-cls", "-obb"
    description:  str    = ""
    notes:        str    = ""


# ── Registry ──────────────────────────────────────────────────────────────────
TASK_REGISTRY: dict[TaskType, TaskConfig] = {
    TaskType.DETECT: TaskConfig(
        task_type    = TaskType.DETECT,
        enabled      = True,
        model_suffix = "",
        description  = "Object Detection — bounding boxes, class, confidence.",
        notes        = "Use yolo26n/s/m/l/x.pt or .onnx.",
    ),
    TaskType.SEGMENT: TaskConfig(
        task_type    = TaskType.SEGMENT,
        enabled      = False,
        model_suffix = "-seg",
        description  = "Instance Segmentation — per-object pixel masks.",
        notes        = "Requires yolo26n-seg.pt export.",
    ),
    TaskType.POSE: TaskConfig(
        task_type    = TaskType.POSE,
        enabled      = False,
        model_suffix = "-pose",
        description  = "Pose Estimation — skeleton keypoints per person.",
        notes        = "Requires yolo26n-pose.pt export.",
    ),
    TaskType.CLASSIFY: TaskConfig(
        task_type    = TaskType.CLASSIFY,
        enabled      = False,
        model_suffix = "-cls",
        description  = "Image Classification — whole-frame top-N labels.",
        notes        = "Requires yolo26n-cls.pt export.",
    ),
    TaskType.OBB: TaskConfig(
        task_type    = TaskType.OBB,
        enabled      = False,
        model_suffix = "-obb",
        description  = "Oriented Bounding Boxes — rotated boxes for aerial imagery.",
        notes        = "Requires yolo26n-obb.pt export.",
    ),
    TaskType.OPEN_VOCAB: TaskConfig(
        task_type    = TaskType.OPEN_VOCAB,
        enabled      = False,
        model_suffix = "",
        description  = "Open-Vocabulary Detection — YOLOE-26, text-prompted.",
        notes        = "Requires yoloe-26s-seg.pt.",
    ),
}


def get_active_tasks() -> list[TaskConfig]:
    return [cfg for cfg in TASK_REGISTRY.values() if cfg.enabled]


def is_task_enabled(task: TaskType) -> bool:
    return TASK_REGISTRY.get(task, TaskConfig(task, False)).enabled


# ── Base task interface ────────────────────────────────────────────────────────
class BaseTask:
    """
    Subclass this to implement a new YOLO26 task.
    The run() method receives a raw frame and returns task-specific results.
    """

    task_type: TaskType = TaskType.DETECT

    def __init__(self, session, class_names: list[str], conf: float = 0.45):
        self._session     = session
        self._class_names = class_names
        self._conf        = conf

    def run(self, frame) -> dict:
        """Run inference on frame, return task-specific result dict."""
        raise NotImplementedError

    def is_enabled(self) -> bool:
        return is_task_enabled(self.task_type)


# ── Active task: ObjectDetection ──────────────────────────────────────────────
class ObjectDetection(BaseTask):
    """
    Object Detection using YOLO26.
    This is the only task currently active in DualVision AI.
    Actual inference is handled by Detector (detector.py).
    This class serves as the canonical task registration point.
    """
    task_type = TaskType.DETECT

    def run(self, frame) -> dict:
        raise NotImplementedError("Use Detector.push_rgb/push_thermal instead.")


# ── Stub tasks (not yet implemented) ──────────────────────────────────────────
class InstanceSegmentation(BaseTask):
    task_type = TaskType.SEGMENT
    def run(self, frame) -> dict:
        raise NotImplementedError("Segmentation not yet enabled.")


class PoseEstimation(BaseTask):
    task_type = TaskType.POSE
    def run(self, frame) -> dict:
        raise NotImplementedError("Pose estimation not yet enabled.")


class ImageClassification(BaseTask):
    task_type = TaskType.CLASSIFY
    def run(self, frame) -> dict:
        raise NotImplementedError("Classification not yet enabled.")


class OrientedBoundingBox(BaseTask):
    task_type = TaskType.OBB
    def run(self, frame) -> dict:
        raise NotImplementedError("OBB not yet enabled.")


class OpenVocabDetection(BaseTask):
    task_type = TaskType.OPEN_VOCAB
    def run(self, frame) -> dict:
        raise NotImplementedError("Open-vocabulary detection not yet enabled.")
