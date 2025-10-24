import warnings
from ultralytics import YOLO, RTDETR
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    model = YOLO(r"ultralytics/cfg/models/mycfg/AMAF-YOLO.yaml")

    model.train(data=r"G:\back_env\AMAF-YOLO\ultralytics\mydata\VisDrone2019\VisDrone2019.yaml",
                epochs=300,
                batch=16,
                workers=8,
                imgsz=640,
                optimizer="SGD",
                cos_lr=True,
                lr0=0.01,
                lrf=0.01,
                # label_smoothing=0.1,
                # close_mosaic=64,
                momentum=0.937,  # 动量设置
                weight_decay=5e-4,  # 权重衰减
                # pretrained=False,  # 加载之前的最佳模型权重
                dropout=0.05,
                # device=0,
                patience=0,
                # name="AI-TOD-RTDETR-L",
                # resume='runs/detect/train6/weights/last.pt'  # 如过想续训就设置 last.pt 的地址
                )
    # print(model)
