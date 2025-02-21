import time
from ultralytics import YOLO
from ultralytics import YOLOWorld
# from ultralytics import RTDETR


if __name__ == '__main__':
    import datetime as dt

    dt_ms = dt.datetime.now().strftime('%Y%m%d%H%M')  # '%Y-%m-%d-%H-%M-%S.%f'
    project_name = 'seg_shimojuan-'
    name = project_name + dt_ms

    task = 'segment'
    project = f'runs/{task}'
    weights = r"D:\code\train\ultralytics\train\models\yolo8s-seg.pt"

    data_cfg = r"D:\code\train\ultralytics\ultralytics\cfg\datasets\VOC_shimojuan.yaml"

    epochs = 100
    batch = 10
    imgsz = 1280
    rect = False
    device = 0
    workers = 8
    pretrained = True
    resume = False
    freeze = 11 # 11
    cache = "disk" # False

    # 加载模型
    # model = YOLO("ultralytics/cfg/models/v8/yolov8s-seg-p6.yaml")
    model = YOLO(weights)  # 加载预训练模型（推荐用于训练）

    # 记录开始时间
    start_time = time.time()

    # 训练模型
    results = model.train(
        task=task,
        imgsz=imgsz,
        rect=rect,
        data=data_cfg,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        pretrained=pretrained,
        resume=resume,
        project=project,
        name=name,
        freeze=freeze,
        cache=cache
    )

    # 记录结束时间
    end_time = time.time()
    # 计算耗时（秒）
    elapsed_time = end_time - start_time

    # 将耗时转换为分钟，并保留两位小数
    elapsed_minutes = round(elapsed_time / 60, 2)

    print(f"耗时: {elapsed_minutes} 分钟")