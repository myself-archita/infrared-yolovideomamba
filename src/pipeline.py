from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import cv2


SPLITS = ("train", "val", "test")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_split_videos(source_root: Path, split: str) -> Iterable[Path]:
    split_dir = source_root / split
    if not split_dir.exists():
        return []
    return split_dir.rglob("infrared.mp4")


def extract_zip_if_needed(archive_path: Path, extract_root: Path) -> Path:
    if extract_root.exists() and any(extract_root.iterdir()):
        return extract_root
    ensure_dir(extract_root)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_root)
    return extract_root


def frame_name(video_stem: str, frame_idx: int) -> str:
    return f"{video_stem}_frame_{frame_idx:06d}.jpg"


def frame_index_from_path(frame_path: Path) -> int:
    stem = frame_path.stem
    try:
        return int(stem.rsplit("_frame_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected frame name: {frame_path.name}") from exc


def sampled_frame_paths(frame_dir: Path, sample_every: int) -> list[Path]:
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if sample_every <= 1:
        return frame_files
    return [p for p in frame_files if frame_index_from_path(p) % sample_every == 0]


def extract_frames_for_split(source_root: Path, output_root: Path, split: str) -> None:
    for video_path in iter_split_videos(source_root, split):
        video_stem = video_path.parent.name
        target_dir = output_root / split / video_stem
        ensure_dir(target_dir)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(target_dir / frame_name(video_stem, idx)), frame)
            idx += 1
        cap.release()


def apply_clahe_bgr(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_train_frames(frames_root: Path, preprocessed_root: Path, sample_every: int = 10) -> None:
    train_root = frames_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Missing train frames at {train_root}")

    for image_path in train_root.rglob("*.jpg"):
        if sample_every > 1 and frame_index_from_path(image_path) % sample_every != 0:
            continue
        rel = image_path.relative_to(train_root)
        target = preprocessed_root / "train" / rel
        ensure_dir(target.parent)

        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img = apply_clahe_bgr(img)
        img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
        cv2.imwrite(str(target), img)


def load_annotation(video_dir: Path) -> dict:
    ann_path = video_dir / "infrared.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    return json.loads(ann_path.read_text(encoding="utf-8"))


def yolo_label_lines(ann: dict, frame_idx: int, width: int, height: int) -> list[str]:
    exist = ann.get("exist", [])
    gt_rect = ann.get("gt_rect", [])
    if frame_idx >= len(exist) or frame_idx >= len(gt_rect) or not exist[frame_idx]:
        return []

    x, y, w, h = gt_rect[frame_idx]
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    bw = w / width
    bh = h / height
    return [f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"]


def convert_annotations_to_yolo_labels(
    source_root: Path, frames_root: Path, labels_root: Path, sample_every: int = 10
) -> None:
    for split in SPLITS:
        for video_dir in (source_root / split).iterdir():
            if not video_dir.is_dir():
                continue
            ann = load_annotation(video_dir)
            frame_dir = frames_root / split / video_dir.name
            if not frame_dir.exists():
                continue

            frame_files = sampled_frame_paths(frame_dir, sample_every)
            first_frame = frame_files[0] if frame_files else None
            if first_frame is None:
                continue
            sample = cv2.imread(str(first_frame))
            if sample is None:
                continue
            height, width = sample.shape[:2]

            target_dir = labels_root / split / video_dir.name
            ensure_dir(target_dir)

            for frame_path in frame_files:
                frame_idx = frame_index_from_path(frame_path)
                lines = yolo_label_lines(ann, frame_idx, width, height)
                label_path = target_dir / f"{frame_path.stem}.txt"
                label_path.write_text("\n".join(lines), encoding="utf-8")


def build_yolo_dataset(
    frames_root: Path, preprocessed_root: Path, labels_root: Path, dataset_root: Path, sample_every: int = 10
) -> None:
    for split in SPLITS:
        src_images = preprocessed_root / "train" if split == "train" else frames_root / split
        src_labels = labels_root / split
        dst_images = dataset_root / split / "images"
        dst_labels = dataset_root / split / "labels"
        ensure_dir(dst_images)
        ensure_dir(dst_labels)

        for video_dir in src_images.iterdir():
            if not video_dir.is_dir():
                continue
            label_dir = src_labels / video_dir.name
            for img_path in sampled_frame_paths(video_dir, sample_every):
                shutil.copy2(img_path, dst_images / img_path.name)
                label_path = label_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, dst_labels / label_path.name)
                else:
                    (dst_labels / label_path.name).write_text("", encoding="utf-8")


def write_dataset_yaml(data_root: Path, output_yaml: Path, class_names: list[str] | None = None) -> None:
    class_names = class_names or ["target"]
    yaml_text = "\n".join(
        [
            f"path: {data_root.as_posix()}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            f"names: {class_names!r}",
            "",
        ]
    )
    output_yaml.write_text(yaml_text, encoding="utf-8")


def train_yolo(model: str, data_yaml: Path, imgsz: int, epochs: int, device: str | None) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Install it first, then rerun training.") from exc

    yolo = YOLO(model)
    kwargs = dict(data=str(data_yaml), imgsz=imgsz, epochs=epochs)
    if device:
        kwargs["device"] = device
    yolo.train(**kwargs)


def run_videomamba(script: Path, args: list[str]) -> None:
    if not script.exists():
        raise FileNotFoundError(
            f"VideoMamba entry script not found: {script}. Point --videomamba-script at your repo's train file."
        )
    subprocess.run([sys.executable, str(script), *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infrared video pipeline")
    parser.add_argument("--archive", type=Path, default=None, help="Path to archive (1).zip")
    parser.add_argument("--source-root", type=Path, default=None, help="Extracted dataset root")
    parser.add_argument("--workdir", type=Path, default=Path("infrared_work"))
    parser.add_argument("--model", type=str, default="yolo26.pt", help="YOLO model checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample-every", type=int, default=10, help="Keep every Nth frame for the YOLO dataset")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--frames-only", action="store_true")
    parser.add_argument("--videomamba-script", type=Path, default=None)
    parser.add_argument("--videomamba-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    source_root = args.source_root
    if source_root is None:
        if args.archive is None:
            raise SystemExit("Provide either --archive or --source-root.")
        source_root = extract_zip_if_needed(args.archive, args.workdir / "extracted")

    frames_root = args.workdir / "frames"
    preprocessed_root = args.workdir / "preprocessed_frames"
    labels_root = args.workdir / "labels"
    dataset_root = args.workdir / "yolo_dataset"
    yaml_path = args.workdir / "dataset.yaml"

    ensure_dir(args.workdir)

    for split in SPLITS:
        extract_frames_for_split(source_root, frames_root, split)

    if args.frames_only:
        return

    preprocess_train_frames(frames_root, preprocessed_root, args.sample_every)
    convert_annotations_to_yolo_labels(source_root, frames_root, labels_root, args.sample_every)
    build_yolo_dataset(frames_root, preprocessed_root, labels_root, dataset_root, args.sample_every)
    write_dataset_yaml(dataset_root, yaml_path)

    if not args.skip_train:
        train_yolo(args.model, yaml_path, args.imgsz, args.epochs, args.device)

    if args.videomamba_script:
        run_videomamba(args.videomamba_script, args.videomamba_args)


if __name__ == "__main__":
    main()

