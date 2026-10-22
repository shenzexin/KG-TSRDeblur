"""Optional Ultralytics training wrapper; supply a valid YOLO dataset YAML."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--model', default='yolov8n.yaml', help='Architecture YAML or user-supplied initial weights')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='0')
    p.add_argument('--output', default='runs/detector')
    a = p.parse_args()
    from ultralytics import YOLO
    # Explicit reference settings. Ultralytics owns the detector loss/augmentations.
    YOLO(a.model).train(data=a.data, imgsz=640, epochs=a.epochs, batch=a.batch_size,
        optimizer='Adam', lr0=1e-3, lrf=1e-3, momentum=0.9, weight_decay=0., cos_lr=True,
        seed=a.seed, device=a.device, project=a.output, name=f'seed{a.seed}')


if __name__ == '__main__':
    main()
