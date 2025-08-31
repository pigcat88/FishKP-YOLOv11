import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('best.pt')
    model.predict(source='datasets/images/val',
                  imgsz=1024,
                  project='runs/predict',
                  name='exp',
                  save=True,
                  save_txt=True

                )
