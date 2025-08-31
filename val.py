import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('best.pt')
    model.val(data='fish_keypoint.yaml',
              split='test',
              imgsz=640,
              batch=16,
              save_json=False,
              project='runs/test',
              name='exp',

              )
