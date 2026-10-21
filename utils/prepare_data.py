import glob
import logging
import os
import random
from typing import Callable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.constants import (
    FRAMES_PER_VIDEO,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_mp_detectors = [
    mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.3
    ),
    mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.3
    ),
]
_mtcnn = MTCNN(keep_all=False, device=_device)

_aug_resize_256 = transforms.Resize((256, 256))
_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

FACE_TRAIN_TRANSFORM = transforms.Compose([
    _aug_resize_256,
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    _to_tensor_norm,
])

FRAME_TRAIN_TRANSFORM = transforms.Compose([
    _aug_resize_256,
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    _to_tensor_norm,
])

VAL_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    _to_tensor_norm,
])


def _detect_face_bbox(
    frame: np.ndarray
) -> Optional[Tuple[float, float, float, float]]:
    """
    Detect the first face in a frame.

    The two MediaPipe detectors (short and full range) are tried first,
    and MTCNN is used as a fallback.

    Args:
        frame (np.ndarray): Frame in BGR format.

    Returns:
        Optional[Tuple[float, float, float, float]]: The face bounding
            box as (x1, y1, x2, y2) in pixels, or None if no face was
            found.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for det in _mp_detectors:
        res = det.process(rgb)
        if res.detections:
            bb = res.detections[0].location_data.relative_bounding_box
            h, w, _ = frame.shape
            return (
                bb.xmin * w,
                bb.ymin * h,
                (bb.xmin + bb.width) * w,
                (bb.ymin + bb.height) * h
            )

    boxes, _ = _mtcnn.detect(rgb)
    if boxes is not None:
        return tuple(boxes[0])
    return None


def _gather_frames(
    video_list: List[Tuple[str, int]],
    require_face: bool
) -> List[Tuple[str, int, int]]:
    """
    Sample frames from each video at a fixed interval.

    Args:
        video_list (List[Tuple[str, int]]): Videos as (path, label)
            pairs, where the label is 0 for real and 1 for fake.
        require_face (bool): If True, only keep frames where a face is
            detected.

    Returns:
        List[Tuple[str, int, int]]: Frames as (video path, frame
            index, label).
    """
    info = []
    for vid_path, label in video_list:
        for idx in sample_frames_fixed_interval(vid_path):
            if not require_face:
                info.append((vid_path, idx, label))
                continue
            cap = cv2.VideoCapture(vid_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            cap.release()
            if ok and crop_face_from_frame(frame) is not None:
                info.append((vid_path, idx, label))
    return info


def _split_videos(
    real_dataset_path: str,
    fake_dataset_path: str,
    tag: str
) -> Tuple[
    List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]
]:
    """
    Shuffle all videos with a fixed seed and split them into train,
    validation and test sets.

    Args:
        real_dataset_path (str): Folder with the real .mp4 videos.
        fake_dataset_path (str): Folder with the fake .mp4 videos.
        tag (str): Name of the pipeline, used in the log messages.

    Returns:
        Tuple[List[Tuple[str, int]], List[Tuple[str, int]],
        List[Tuple[str, int]]]: The train, validation and test videos
            as (path, label) pairs.
    """
    random.seed(RANDOM_SEED)
    real_videos = sorted(glob.glob(os.path.join(real_dataset_path, "*.mp4")))
    fake_videos = sorted(glob.glob(os.path.join(fake_dataset_path, "*.mp4")))

    vids = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]
    random.shuffle(vids)

    n_total = len(vids)
    n_train, n_val = int(TRAIN_RATIO * n_total), int(VAL_RATIO * n_total)
    train_v = vids[:n_train]
    val_v = vids[n_train:n_train + n_val]
    test_v = vids[n_train + n_val:]
    logging.info(
        f"[{tag}] split: {len(train_v)} train / {len(val_v)} val / "
        f"{len(test_v)} test"
    )
    return train_v, val_v, test_v


def crop_face_from_frame(
    frame: np.ndarray,
    fallback_size: Tuple[int, int] = (224, 224),
    margin: float = 0.2
) -> Optional[Image.Image]:
    """
    Crop the face from a frame, with a margin around it.

    Args:
        frame (np.ndarray): Frame in BGR format.
        fallback_size (Tuple[int, int]): Size the crop is resized to.
        margin (float): Margin added on each side, as a fraction of the
            face width and height.

    Returns:
        Optional[Image.Image]: The face crop, or None if no face was
            found.
    """
    bbox = _detect_face_bbox(frame)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape
    dw, dh = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(x1 - dw, 0)), int(max(y1 - dh, 0))
    x2, y2 = int(min(x2 + dw, w)), int(min(y2 + dh, h))

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return Image.fromarray(cv2.resize(face, fallback_size))


def sample_frames_fixed_interval(
    video_path: str,
    frames_per_video: int = FRAMES_PER_VIDEO
) -> List[int]:
    """
    Get the indices of frames sampled at a fixed interval from a video.

    Args:
        video_path (str): Path to the video.
        frames_per_video (int): Number of frames to sample.

    Returns:
        List[int]: The frame indices, or an empty list if the video has
            no frames.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return []
    step = max(total // frames_per_video, 1)
    return [i * step for i in range(frames_per_video) if i * step < total]


class DeepFakeFrameDataset(Dataset):
    """
    Dataset of video frames that returns (tensor_image, label,
    video_path).

    If face_crop is True, it tries to crop the face; otherwise it uses
    the whole frame.
    """

    def __init__(
        self,
        frame_info_list: List[Tuple[str, int, int]],
        transform: Optional[Callable] = None,
        face_crop: bool = True
    ) -> None:
        """
        Initialize the dataset.

        Args:
            frame_info_list (List[Tuple[str, int, int]]): Frames as
                (video path, frame index, label).
            transform (Optional[Callable]): Transform applied to each
                image.
            face_crop (bool): Whether to crop the face from each frame.

        Returns:
            None
        """
        self.frame_info_list = frame_info_list
        self.transform = transform
        self.face_crop = face_crop


    def __len__(
        self
    ) -> int:
        """
        Get the number of frames in the dataset.

        Returns:
            int: The number of frames.
        """
        return len(self.frame_info_list)


    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load one frame.

        If no face is found, up to 5 random frames of the same video are
        tried. If all fail, the whole frame resized to 224x224 is used.

        Args:
            idx (int): Index of the frame.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: The transformed
                image, its label and the path of its video.
        """
        video_path, frame_num, label = self.frame_info_list[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_pil, attempt, ok = None, 0, False
        while attempt < 5 and frame_pil is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()
            if not ok:
                break

            if self.face_crop:
                frame_pil = crop_face_from_frame(frame)
            else:
                frame_pil = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

            if frame_pil is None:
                frame_num = random.randint(0, max(total_frames - 1, 0))
                attempt += 1
        cap.release()

        if frame_pil is None and ok:
            frame_pil = Image.fromarray(cv2.resize(frame, (224, 224)))

        if self.transform:
            frame_pil = self.transform(frame_pil)
        return frame_pil, torch.tensor(label, dtype=torch.long), video_path


def prepare_data_faces(
    real_dataset_path: str,
    fake_dataset_path: str
) -> Tuple[DeepFakeFrameDataset, DeepFakeFrameDataset, DeepFakeFrameDataset]:
    """
    Build the face-cropped frame datasets.

    Only frames where a face is detected are kept.

    Args:
        real_dataset_path (str): Folder with the real .mp4 videos.
        fake_dataset_path (str): Folder with the fake .mp4 videos.

    Returns:
        Tuple[DeepFakeFrameDataset, DeepFakeFrameDataset,
        DeepFakeFrameDataset]: The train, validation and test datasets.
    """
    train_v, val_v, test_v = _split_videos(
        real_dataset_path, fake_dataset_path, "faces"
    )

    train_info = _gather_frames(train_v, require_face=True)
    val_info = _gather_frames(val_v, require_face=True)
    test_info = _gather_frames(test_v, require_face=True)
    logging.info(
        f"[faces] frames: train={len(train_info)} val={len(val_info)} "
        f"test={len(test_info)}"
    )

    train_ds = DeepFakeFrameDataset(
        train_info, FACE_TRAIN_TRANSFORM, face_crop=True
    )
    val_ds = DeepFakeFrameDataset(
        val_info, VAL_TEST_TRANSFORM, face_crop=True
    )
    test_ds = DeepFakeFrameDataset(
        test_info, VAL_TEST_TRANSFORM, face_crop=True
    )
    return train_ds, val_ds, test_ds


def prepare_data_full_frame(
    real_dataset_path: str,
    fake_dataset_path: str
) -> Tuple[DeepFakeFrameDataset, DeepFakeFrameDataset, DeepFakeFrameDataset]:
    """
    Build the full-frame datasets (no face detection, faster).

    Args:
        real_dataset_path (str): Folder with the real .mp4 videos.
        fake_dataset_path (str): Folder with the fake .mp4 videos.

    Returns:
        Tuple[DeepFakeFrameDataset, DeepFakeFrameDataset,
        DeepFakeFrameDataset]: The train, validation and test datasets.
    """
    train_v, val_v, test_v = _split_videos(
        real_dataset_path, fake_dataset_path, "frames"
    )

    train_info = _gather_frames(train_v, require_face=False)
    val_info = _gather_frames(val_v, require_face=False)
    test_info = _gather_frames(test_v, require_face=False)
    logging.info(
        f"[frames] frames: train={len(train_info)} val={len(val_info)} "
        f"test={len(test_info)}"
    )

    train_ds = DeepFakeFrameDataset(
        train_info, FRAME_TRAIN_TRANSFORM, face_crop=False
    )
    val_ds = DeepFakeFrameDataset(
        val_info, VAL_TEST_TRANSFORM, face_crop=False
    )
    test_ds = DeepFakeFrameDataset(
        test_info, VAL_TEST_TRANSFORM, face_crop=False
    )
    return train_ds, val_ds, test_ds
