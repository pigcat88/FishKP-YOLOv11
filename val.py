import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    #model = YOLO('runs/train-pose/exp3/weights/best.pt')
    model = YOLO('runs/train-pose/exp65/weights/best.pt')
    model.val(data='fish_keypoint.yaml',
              split='test',
              imgsz=640,
              batch=16,
              #crop_fraction = 1.0,
              #iou=0.7,
              #conf=0.5,
              #rect=False,
              save_json=False,
              project='runs/val',
              name='exp',
              )