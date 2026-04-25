import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from easymocap.smplmodel import select_nf


T = TypeVar("T")


@dataclass(frozen=True)
class _SequenceSource:
    scene_name: str
    session_name: str
    sequence_name: str
    seq_id: int
    start_frame: int
    end_frame: int
    params_path: Path
    keypoints3d_dir: Path


@dataclass(frozen=True)
class _FrameRecord:
    frame_id: int
    mask: float
    pose: np.ndarray
    Rh: np.ndarray
    Th: np.ndarray
    shape: np.ndarray


@dataclass(frozen=True)
class _ClipRecord:
    sample_name: str
    session_name: str
    seq_id: int
    frame_ids: Tuple[int, ...]
    times: np.ndarray
    mask: np.ndarray
    pose: np.ndarray
    Rh: np.ndarray
    Th: np.ndarray
    shape: np.ndarray
    motion: np.ndarray


class GigaHandDataset(Dataset):
    """
    Loads right-hand MANO parameter sequences from GigaHands.

    The implementation follows the same data flow as the provided
    `render_mesh_video.py`:
    1. read `hand_poses/<session>/params/<seq>.json`
    2. read `hand_poses/<session>/keypoints_3d/<seq>/chosen_frames_right.json`
    3. use EasyMocap's `select_nf` to extract per-frame MANO params

    Each motion vector is `[Rh_6d, pose_6d, Th]`. Both `Rh` and the MANO hand
    pose parameters are converted from axis-angle to a 6D rotation
    representation before being returned or concatenated into motion. MANO
    shape coefficients are returned separately and are not included in the
    motion state.
    """

    def __init__(
        self,
        dataset_root: str,
        split: Optional[str] = None,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        text_file: Optional[str] = None,
        random_mask: bool = False,
        random_mask_prob: float = 0.15,
        fps: float = 30.0,
        max_sequences: Optional[int] = None,
        history_len: int = 16,
        horizon: int = 0,
        time_stride_aug_max: int = 1,
        verbose: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.hand_pose_root = self.dataset_root / "hand_poses"
        self.text_file = (
            Path(text_file).expanduser().resolve()
            if text_file is not None
            else self.dataset_root / "annotations_v2.jsonl"
        )
        self.split = split
        self.random_mask = random_mask
        self.random_mask_prob = float(random_mask_prob)
        self.fps = float(fps)
        self.max_sequences = max_sequences
        self.history_len = int(history_len)
        self.current_horizon = int(horizon)
        self.time_stride_aug_max = int(time_stride_aug_max)
        self.verbose = verbose
        self.return_full_sequence = split in {"val", "test"}

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        if not self.text_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.text_file}")
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        if self.current_horizon < 0:
            raise ValueError("horizon must be non-negative")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not 0.0 <= self.random_mask_prob <= 1.0:
            raise ValueError("random_mask_prob must be in [0, 1]")
        if self.time_stride_aug_max <= 0:
            raise ValueError("time_stride_aug_max must be positive")
        if split is not None and split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of None/train/val/test/all")
        if len(split_ratio) != 3 or any(r < 0 for r in split_ratio) or sum(split_ratio) <= 0:
            raise ValueError("split_ratio must contain three non-negative numbers")

        sources = self._discover_sources()
        if self.max_sequences is not None:
            sources = sources[: self.max_sequences]
        if not sources:
            raise RuntimeError(f"No GigaHands sequences found from {self.text_file}")

        self.sources = self._apply_split(sources)
        records = self._build_records(self.sources)
        if not records:
            raise RuntimeError("No valid right-hand sequences could be constructed")

        self.all_records = records
        self.records = records
        self.train_sampling_options: Optional[List[np.ndarray]] = None
        self._applied_training_horizon: Optional[int] = None
        if not self.return_full_sequence:
            self.set_training_horizon(self.current_horizon)

        if not self.records:
            raise RuntimeError(f"Split '{self.split}' produced zero sequences")

        first = self.records[0]
        self.motion_dim = int(first.motion.shape[-1])
        self.pose_dim = int(first.pose.shape[-1])
        self.shape_dim = int(first.shape.shape[-1])

    def __len__(self) -> int:
        return len(self.records)

    def set_training_horizon(self, horizon: int) -> None:
        horizon = int(horizon)
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        self.current_horizon = horizon
        if self.return_full_sequence:
            return
        if self._applied_training_horizon == horizon and self.train_sampling_options is not None:
            return

        filtered_records: List[_ClipRecord] = []
        sampling_options: List[np.ndarray] = []
        for record in self.all_records:
            options = self._valid_sampling_options(
                record.mask,
                target_len=self.history_len + self.current_horizon,
                max_stride=self.time_stride_aug_max,
            )
            if options.size == 0:
                continue
            filtered_records.append(record)
            sampling_options.append(options)
        if not filtered_records:
            raise RuntimeError(
                f"No valid training windows found for history_len={self.history_len}, "
                f"horizon={self.current_horizon}, max_stride={self.time_stride_aug_max}."
            )
        self.records = filtered_records
        self.train_sampling_options = sampling_options
        self._applied_training_horizon = horizon

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        if self.return_full_sequence:
            clip = record
        else:
            if self.train_sampling_options is None:
                raise RuntimeError("Training sampling options were not initialized.")
            clip = self._sample_training_window(record, self.train_sampling_options[index])
        Th = clip.Th.copy()

        motion = self._compose_motion(clip.pose, clip.Rh, Th)
        mask = clip.mask.copy().astype(np.float32)
        if self.random_mask:
            mask = self._apply_random_mask(mask)
        return {
            "motion": torch.from_numpy(motion).float(),
            "mask": torch.from_numpy(mask).float(),
            "pose": torch.from_numpy(clip.pose.copy()).float(),
            "Rh": torch.from_numpy(clip.Rh.copy()).float(),
            "Th": torch.from_numpy(Th).float(),
            "shape": torch.from_numpy(clip.shape.copy()).float(),
            "frame_ids": torch.tensor(clip.frame_ids, dtype=torch.long),
            "times": torch.from_numpy(clip.times.copy()).float(),
            "session_name": clip.session_name,
            "seq_id": clip.seq_id,
            "sample_name": clip.sample_name,
        }

    def __repr__(self) -> str:
        return (
            f"GigaHandDataset(num_sequences={len(self)}, history_len={self.history_len}, "
            f"horizon={self.current_horizon}, motion_dim={self.motion_dim}, split={self.split!r}, "
            f"full_sequence={self.return_full_sequence}, root='{self.dataset_root}')"
        )

    def _discover_sources(self) -> List[_SequenceSource]:
        sources: List[_SequenceSource] = []
        with self.text_file.open("r", encoding="utf-8") as file:
            for line in file:
                script_info = json.loads(line)
                sequence = script_info["sequence"]
                sequence_name = sequence[0] if isinstance(sequence, list) else sequence
                scene_name = script_info["scene"]
                start_frame = int(script_info["start_frame_id"])
                end_frame = int(script_info["end_frame_id"])
                script_text = script_info.get("clarify_annotation", "None")

                if self.split != "all" and script_text in {"None", "Buggy"}:
                    continue

                resolved = self._resolve_sequence_paths(scene_name, sequence_name)
                if resolved is None:
                    continue
                params_path, keypoints3d_dir, session_name, seq_id = resolved

                sources.append(
                    _SequenceSource(
                        scene_name=scene_name,
                        session_name=session_name,
                        sequence_name=sequence_name,
                        seq_id=seq_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        params_path=params_path,
                        keypoints3d_dir=keypoints3d_dir,
                    )
                )
        return sources

    def _build_records(self, sources: List[_SequenceSource]) -> List[_ClipRecord]:
        iterator: Iterable[_SequenceSource] = sources
        if self.verbose:
            iterator = tqdm(sources, desc="Loading GigaHands", unit="seq")

        records: List[_ClipRecord] = []
        for source in iterator:
            frames = self._load_right_hand_frames(source)
            if len(frames) == 0:
                continue
            records.append(self._frames_to_record(source, frames))
        return records

    def _load_right_hand_frames(self, source: _SequenceSource) -> List[_FrameRecord]:
        with source.params_path.open("r", encoding="utf-8") as f:
            mano_params = json.load(f)

        if "right" not in mano_params:
            return []
        params_right = {key: np.asarray(value) for key, value in mano_params["right"].items()}

        chosen_right = self._read_frame_list(source.keypoints3d_dir / "chosen_frames_right.json")
        chosen_left_path = source.keypoints3d_dir / "chosen_frames_left.json"
        chosen_left = self._read_frame_list(chosen_left_path) if chosen_left_path.exists() else []
        union_frames = sorted(set(chosen_left) | set(chosen_right))

        n_param_frames = int(params_right["poses"].shape[0])
        if n_param_frames == len(union_frames):
            param_indices = [union_frames.index(frame_id) for frame_id in chosen_right]
        elif n_param_frames == len(chosen_right):
            param_indices = list(range(len(chosen_right)))
        else:
            raise ValueError(
                f"Frame count mismatch for {source.params_path}: "
                f"right={len(chosen_right)}, union={len(union_frames)}, params={n_param_frames}"
            )

        valid_frames: List[_FrameRecord] = []
        for frame_id, param_index in zip(chosen_right, param_indices):
            param_right = select_nf(params_right, param_index)
            pose = self._pose_axis_angle_to_rot6d(
                np.asarray(param_right["poses"], dtype=np.float32).reshape(-1)
            )
            Rh = self._axis_angle_to_rot6d(
                np.asarray(param_right["Rh"], dtype=np.float32).reshape(-1)
            )
            Th = np.asarray(param_right["Th"], dtype=np.float32).reshape(-1)
            shape = np.asarray(param_right["shapes"], dtype=np.float32).reshape(-1)
            valid_frames.append(
                _FrameRecord(frame_id=frame_id, mask=1.0, pose=pose, Rh=Rh, Th=Th, shape=shape)
            )
        valid_frames.sort(key=lambda item: item.frame_id)
        start_frame = source.start_frame
        end_frame = source.end_frame
        if end_frame == -1:
            valid_frames = [frame for frame in valid_frames if frame.frame_id >= start_frame]
        else:
            valid_frames = [
                frame
                for frame in valid_frames
                if start_frame <= frame.frame_id <= end_frame
            ]
        return self._densify_frames(valid_frames, start_frame=start_frame, end_frame=end_frame)

    def _frames_to_record(
        self, source: _SequenceSource, frames: List[_FrameRecord]
    ) -> _ClipRecord:
        pose = np.stack([item.pose for item in frames], axis=0).astype(np.float32)
        Rh = np.stack([item.Rh for item in frames], axis=0).astype(np.float32)
        Th = np.stack([item.Th for item in frames], axis=0).astype(np.float32)
        shape = np.stack([item.shape for item in frames], axis=0).astype(np.float32)
        mask = np.asarray([item.mask for item in frames], dtype=np.float32)
        frame_ids = tuple(item.frame_id for item in frames)
        times = (np.asarray(frame_ids, dtype=np.float32) - float(frame_ids[0])) / self.fps
        sample_name = f"{source.scene_name}/{source.sequence_name}:{frame_ids[0]}-{frame_ids[-1]}"
        return _ClipRecord(
            sample_name=sample_name,
            session_name=source.session_name,
            seq_id=source.seq_id,
            frame_ids=frame_ids,
            times=times,
            mask=mask,
            pose=pose,
            Rh=Rh,
            Th=Th,
            shape=shape,
            motion=self._compose_motion(pose, Rh, Th),
        )

    @staticmethod
    def _valid_sampling_options(mask: np.ndarray, target_len: int, max_stride: int) -> np.ndarray:
        if target_len <= 0:
            raise ValueError("target_len must be positive.")
        if max_stride <= 0:
            raise ValueError("max_stride must be positive.")

        valid = mask > 0
        target_offsets = np.arange(target_len, dtype=np.int64)
        options: List[np.ndarray] = []
        total_len = mask.shape[0]
        for stride in range(1, max_stride + 1):
            span = 1 + (target_len - 1) * stride
            if span > total_len:
                break
            starts = np.arange(total_len - span + 1, dtype=np.int64)
            idx = starts[:, None] + stride * target_offsets[None, :]
            is_valid = valid[idx].all(axis=1)
            valid_starts = starts[is_valid]
            if valid_starts.size == 0:
                continue
            stride_column = np.full((valid_starts.shape[0], 1), stride, dtype=np.int64)
            options.append(np.concatenate([valid_starts[:, None], stride_column], axis=1))

        if not options:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(options, axis=0)

    def _sample_training_window(self, record: _ClipRecord, sampling_options: np.ndarray) -> _ClipRecord:
        target_len = self.history_len + self.current_horizon
        if target_len <= 0:
            raise RuntimeError("Target training window length must be positive.")
        if sampling_options.size == 0:
            raise RuntimeError(
                f"Sequence {record.sample_name} has no valid sampling options for target length {target_len}."
            )
        choice = sampling_options[np.random.randint(0, sampling_options.shape[0])]
        start = int(choice[0])
        stride = int(choice[1])
        idx = start + stride * np.arange(target_len, dtype=np.int64)

        pose = record.pose[idx].copy()
        Rh = record.Rh[idx].copy()
        Th = record.Th[idx].copy()
        shape = record.shape[idx].copy()
        mask = record.mask[idx].copy()
        frame_ids = tuple(record.frame_ids[i] for i in idx.tolist())
        times = record.times[idx].copy()
        times = times - times[0]

        return _ClipRecord(
            sample_name=record.sample_name,
            session_name=record.session_name,
            seq_id=record.seq_id,
            frame_ids=frame_ids,
            times=times.astype(np.float32),
            mask=mask.astype(np.float32),
            pose=pose.astype(np.float32),
            Rh=Rh.astype(np.float32),
            Th=Th.astype(np.float32),
            shape=shape.astype(np.float32),
            motion=self._compose_motion(pose, Rh, Th),
        )

    def _apply_random_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.return_full_sequence:
            return mask
        history_len = self.history_len
        target_len = history_len + self.current_horizon
        if mask.shape[0] != target_len:
            raise RuntimeError(
                f"Random-mask expects length {target_len}, got {mask.shape[0]}."
            )

        masked = mask.copy()
        if history_len > 2:
            candidate = np.arange(1, history_len - 1, dtype=np.int64)
            drop = np.random.rand(candidate.shape[0]) < self.random_mask_prob
            masked[candidate[drop]] = 0.0

        # Keep the history anchor and all future supervision valid.
        masked[0] = mask[0]
        masked[history_len - 1] = mask[history_len - 1]
        masked[history_len:] = mask[history_len:]
        return masked.astype(np.float32)

    def _compose_motion(
        self,
        pose: np.ndarray,
        Rh: np.ndarray,
        Th: np.ndarray,
    ) -> np.ndarray:
        parts = [Rh, pose, Th]
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _apply_split(self, items: List[T]) -> List[T]:
        if self.split is None or self.split == "all":
            return items

        if len(items) < 2:
            raise ValueError(
                f"Cannot create split='{self.split}' from only {len(items)} sequence(s). "
                "Use split='all' or increase max_sequences / available data."
            )

        train_items, remaining_items = train_test_split(
            items, test_size=0.2, random_state=42
        )
        if len(remaining_items) < 2:
            raise ValueError(
                f"Cannot create validation/test subsets from only {len(remaining_items)} held-out sequence(s) "
                f"after the first split (total sequences: {len(items)}). "
                "Use split='all' or increase max_sequences / available data."
            )
        test_items, val_items = train_test_split(
            remaining_items, test_size=0.25, random_state=42
        )

        if self.split == "train":
            return train_items
        if self.split == "val":
            return val_items
        return test_items

    @staticmethod
    def _read_frame_list(path: Path) -> List[int]:
        with path.open("r", encoding="utf-8") as f:
            values = json.load(f)
        return sorted(int(v) for v in values)

    def _resolve_sequence_paths(
        self, scene_name: str, sequence_name: str
    ) -> Optional[Tuple[Path, Path, str, int]]:
        if not self.hand_pose_root.exists():
            return None

        session_dir = self.hand_pose_root / scene_name
        params_path = session_dir / "params" / f"{sequence_name}.json"
        keypoints_dir = session_dir / "keypoints_3d" / sequence_name
        if params_path.exists() and (keypoints_dir / "chosen_frames_right.json").exists():
            seq_id = int(sequence_name) if sequence_name.isdigit() else -1
            return params_path, keypoints_dir, scene_name, seq_id
        return None

    @staticmethod
    def _densify_frames(
        valid_frames: List[_FrameRecord], start_frame: int, end_frame: int
    ) -> List[_FrameRecord]:
        if not valid_frames:
            return []

        frame_map = {frame.frame_id: frame for frame in valid_frames}
        first_valid = valid_frames[0]
        pose_zeros = np.zeros_like(first_valid.pose)
        Rh_zeros = np.zeros_like(first_valid.Rh)
        Th_zeros = np.zeros_like(first_valid.Th)
        shape_zeros = np.zeros_like(first_valid.shape)

        dense_end = end_frame if end_frame != -1 else valid_frames[-1].frame_id
        dense_frames: List[_FrameRecord] = []
        for frame_id in range(start_frame, dense_end + 1):
            if frame_id in frame_map:
                dense_frames.append(frame_map[frame_id])
            else:
                dense_frames.append(
                    _FrameRecord(
                        frame_id=frame_id,
                        mask=0.0,
                        pose=pose_zeros.copy(),
                        Rh=Rh_zeros.copy(),
                        Th=Th_zeros.copy(),
                        shape=shape_zeros.copy(),
                    )
                )
        return dense_frames

    @staticmethod
    def _axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
        rotmat = Rotation.from_rotvec(axis_angle.reshape(1, 3)).as_matrix()[0]
        # Project convention: flatten `rotmat[:, :2]` in NumPy/C row-major order,
        # i.e. [r00, r01, r10, r11, r20, r21]. This is different from the more
        # common column-stacked 6D layout [r00, r10, r20, r01, r11, r21], so the
        # matching decoder in `torch_ode2vae_hand._rot6d_to_axis_angle()` must
        # keep this exact ordering for checkpoints/evaluation to stay consistent.
        return rotmat[:, :2].reshape(-1).astype(np.float32)

    @staticmethod
    def _pose_axis_angle_to_rot6d(pose_axis_angle: np.ndarray) -> np.ndarray:
        if pose_axis_angle.size % 3 != 0:
            raise ValueError(
                f"Expected pose axis-angle vector length to be divisible by 3, got {pose_axis_angle.size}"
            )
        joint_axis_angles = pose_axis_angle.reshape(-1, 3)
        rotmats = Rotation.from_rotvec(joint_axis_angles).as_matrix()
        # Same row-major flattening convention as `_axis_angle_to_rot6d()`.
        return rotmats[:, :, :2].reshape(-1).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick sanity check for GigaHandDataset.")
    parser.add_argument("dataset_root", type=str, help="Path to the GigaHands dataset root.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--text-file", type=str, default=None, dest="text_file")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--index", type=int, default=0, help="Sample index to inspect.")
    parser.add_argument("--random-mask", action="store_true")
    parser.add_argument("--random-mask-prob", type=float, default=0.15)
    parser.add_argument("--quiet", action="store_true", help="Disable tqdm loading output.")
    args = parser.parse_args()

    dataset = GigaHandDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        text_file=args.text_file,
        random_mask=args.random_mask,
        random_mask_prob=args.random_mask_prob,
        fps=args.fps,
        history_len=args.history_len,
        horizon=args.horizon,
        verbose=not args.quiet,
    )

    print(dataset)
    print(f"num_sources={len(dataset.sources)}")
    print(f"num_sequences={len(dataset)}")
    print(f"motion_dim={dataset.motion_dim}, pose_dim={dataset.pose_dim}, shape_dim={dataset.shape_dim}")

    if len(dataset) == 0:
        raise RuntimeError("Dataset loaded successfully but contains no sequences.")

    index = max(0, min(args.index, len(dataset) - 1))
    sample = dataset[index]
    print(f"inspect_index={index}")
    print(f"sample_name={sample['sample_name']}")
    print(f"session_name={sample['session_name']}, seq_id={sample['seq_id']}")
    print(f"motion.shape={tuple(sample['motion'].shape)}")
    print(f"mask.shape={tuple(sample['mask'].shape)}")
    print(f"pose.shape={tuple(sample['pose'].shape)}")
    print(f"Rh.shape={tuple(sample['Rh'].shape)}")
    print(f"Th.shape={tuple(sample['Th'].shape)}")
    print(f"shape.shape={tuple(sample['shape'].shape)}")
    print(f"frame_ids[:5]={sample['frame_ids'][:5].tolist()}")
    print(f"times[:5]={sample['times'][:5].tolist()}")
    print(f"mask[:10]={sample['mask'][:10].tolist()}")
