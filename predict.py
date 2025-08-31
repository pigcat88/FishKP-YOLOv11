import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/train-pose/exp28/weights/best.pt')
    model.predict(source='datasets/images/val',
                  imgsz=1024,
                  project='runs/predict',
                  name='exp',
                  save=True,
                  #conf=0.5,
                  #iou=0.5,
                  save_txt=True
                )