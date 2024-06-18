import argparse
import csv
from pathlib import Path
import torch

from utils.general import non_max_suppression
from utils.dataloaders import LoadImages
from models.common import DetectMultiBackend

def run(weights, source, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, classes=None):
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names

    # Load image
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)

    # Run inference
    model.warmup()
    for path, im, im0s, _, _ in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes)[0]

        # Process predictions
        for *xyxy, conf, cls in reversed(pred):
            label = names[int(cls)]
            confidence = float(conf)
            print(f"Signature: {label}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    weights = "path/to/your/yolo5_model.pt"  # Replace with the path to your trained model
    source = "path/to/your/photo.jpg"  # Replace with the path to the photo you want to test
    run(weights, source)

#How to run:   python detect.py --weights path/to/your/yolo5_model.pt --source path/to/your/photo.jpg --output_dir output
