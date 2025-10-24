from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs\detect\VD\VD-AMAF-YOLO\weights\best.pt')  # 自己训练结束后的模型权重
    model.val(data="ultralytics\mydata\VisDrone2019\VisDrone2019.yaml",
              split='val',
              imgsz=448,
              batch=16,
              conf=0.001,  # 置信度阈值
              iou=0.6,  # NMS时IoU的阈值。
              max_det=300,
              half=False,
              save_json=False,  # if you need to cal coco metrice
              device="cuda",
              plots=False,
              rect=True,
              name="neu-yolo",
              verbose=False,  # 是否打印出每个类别的mAP
              workers=0,
              augment=False,
              agnostic_nms=False,
              single_cls=False
              )

