import os


def train_yolo(data):
    print('Training YOLO Version 5!')
    os.system(f"python yolov5/train.py --data {data} --cfg yolov5s.yaml --batch-size 8 --name Model --epochs 100")


def export_model():
    os.system('!python yolov5/export.py --weights yolov5/runs/train/Model2/weights/best.pt --include torchscript onnx ')


if __name__ == '__main__':
    train_yolo('static/datasets/images-Set3-yolo/data.yaml')
    export_model()
