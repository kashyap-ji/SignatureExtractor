import argparse
import csv
from pathlib import Path
import torch
import cv2

from utils.general import non_max_suppression
from utils.dataloaders import LoadImages
from models.common import DetectMultiBackend

def run(weights, source, output_dir, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, classes=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.warmup()
    for path, im, im0s, _, _ in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes)[0]

        img_name = Path(path).stem
        for *xyxy, conf, cls in reversed(pred):
            label = names[int(cls)]
            confidence = float(conf)
            print(f"Signature: {label}, Confidence: {confidence:.2f}")

            # Save extracted signature
            x1, y1, x2, y2 = map(int, xyxy)
            signature = im0s[y1:y2, x1:x2]
            save_path = output_dir / f"{img_name}_{label}.jpg"
            cv2.imwrite(str(save_path), signature)

if __name__ == "__main__":
    weights = "path/to/your/yolo5_model.pt"
    source = "path/to/your/photo.jpg"
    output_dir = Path("output")
    run(weights, source, output_dir)
