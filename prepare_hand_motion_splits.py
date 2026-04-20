import argparse
import json
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter GigaHands sequences by 3D joint motion and create train/val/test splits."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the GigaHands dataset root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save filtered split files",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help="JSONL file to read sequence metadata from (default: <dataset_root>/annotations_v2.jsonl)",
    )
    parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=30,
        help="Minimum number of valid right-hand frames required",
    )
    parser.add_argument(
        "--min-motion-mm",
        type=float,
        default=20.0,
        help="Minimum peak wrist-relative joint motion in millimeters",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_right_hand_joints(keypoints_dir):
    chosen_path = keypoints_dir / "chosen_frames_right.json"
    joints_path = keypoints_dir / "right.jsonl"
    if not chosen_path.exists() or not joints_path.exists():
        return None, None

    with chosen_path.open("r", encoding="utf-8") as f:
        frame_ids = np.asarray(json.load(f), dtype=np.int64)
    with joints_path.open("r", encoding="utf-8") as f:
        joints = np.asarray([json.loads(line) for line in f if line.strip()], dtype=np.float32)

    if len(frame_ids) != len(joints) or joints.ndim != 3 or joints.shape[1] == 0:
        return None, None
    return frame_ids, joints[..., :3]


def select_clip(frame_ids, joints, start_frame, end_frame):
    mask = frame_ids >= int(start_frame)
    if int(end_frame) != -1:
        mask &= frame_ids <= int(end_frame)
    return frame_ids[mask], joints[mask]


def compute_motion_mm(joints):
    local_joints = joints - joints[:, :1]
    first_frame = local_joints[:1]
    frame_motion = np.linalg.norm(local_joints - first_frame, axis=-1).mean(axis=-1)
    return float(frame_motion.max() * 1000.0)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    annotation_file = (
        args.annotation_file.expanduser().resolve()
        if args.annotation_file is not None
        else dataset_root / "annotations_v2.jsonl"
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Require train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1")

    kept = []
    scanned = 0
    for item in load_jsonl(annotation_file):
        scanned += 1
        scene = item["scene"]
        sequence = item["sequence"][0] if isinstance(item["sequence"], list) else item["sequence"]
        keypoints_dir = dataset_root / "hand_poses" / scene / "keypoints_3d" / str(sequence)

        frame_ids, joints = load_right_hand_joints(keypoints_dir)
        if frame_ids is None:
            continue

        clip_frame_ids, clip_joints = select_clip(
            frame_ids,
            joints,
            item.get("start_frame_id", 0),
            item.get("end_frame_id", -1),
        )
        if len(clip_frame_ids) < args.min_valid_frames:
            continue

        motion_mm = compute_motion_mm(clip_joints)
        if motion_mm < args.min_motion_mm:
            continue

        row = dict(item)
        row["valid_frame_count"] = int(len(clip_frame_ids))
        row["motion_mm"] = round(motion_mm, 3)
        kept.append(row)

    random.Random(args.seed).shuffle(kept)

    n_total = len(kept)
    n_train = int(round(n_total * args.train_ratio))
    n_val = int(round(n_total * args.val_ratio))
    n_train = min(n_train, n_total)
    n_val = min(n_val, max(0, n_total - n_train))
    n_test = n_total - n_train - n_val

    train_rows = kept[:n_train]
    val_rows = kept[n_train : n_train + n_val]
    test_rows = kept[n_train + n_val :]

    write_jsonl(output_dir / "all_filtered.jsonl", kept)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    summary = {
        "dataset_root": str(dataset_root),
        "annotation_file": str(annotation_file),
        "min_valid_frames": int(args.min_valid_frames),
        "min_motion_mm": float(args.min_motion_mm),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "scanned_sequences": int(scanned),
        "kept_sequences": int(n_total),
        "train_sequences": int(len(train_rows)),
        "val_sequences": int(len(val_rows)),
        "test_sequences": int(len(test_rows)),
        "motion_mm_mean": round(float(np.mean([row["motion_mm"] for row in kept])) if kept else 0.0, 3),
        "motion_mm_median": round(float(np.median([row["motion_mm"] for row in kept])) if kept else 0.0, 3),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
