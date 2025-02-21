from ultralytics import YOLO
from ultralytics import YOLOWorld
# from ultralytics import RTDETR

if __name__ == '__main__':
    import datetime as dt

    dt_ms = dt.datetime.now().strftime('%Y%m%d%H%M')  # '%Y-%m-%d-%H-%M-%S.%f'
    project_name = '思泉石墨卷_new_'
    name = project_name + dt_ms

    task = 'segment'
    project = f'runs/{task}'
    weights = r"runs/segment/思泉石墨卷_new_2024062917302/weights/last.pt"

    data_cfg = r"dataset/20240626_思泉石墨卷_new_seg.yaml"

    epochs = 10
    batch = 2
    imgsz = 1280
    rect = False
    device = 0
    workers = 8
    pretrained = False
    resume = False
    freeze = 10
    cache = False

    # 加载模型
    model = YOLO("ultralytics/cfg/models/v8/yolov8s-seg-p6.yaml")
    # model = YOLO(weights)  # 加载预训练模型（推荐用于训练）

    # 自定义冻结层
    # def freeze_model(trainer):
    ## Retrieve the batch data
    # model = trainer.model
    # print('Befor Freeze')
    # for k, v in model.named_parameters():
    #    print('\t', k,'\t', v.requires_grad)
    #
    #
    # freeze = 10
    # freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    # for k, v in model.named_parameters():
    #    v.requires_grad = True  # train all layers
    #    if any(x in k for x in freeze):
    #        print(f'freezing {k}')
    #        v.requires_grad = False
    # print('After Freeze')
    # for k, v in model.named_parameters():
    #    print('\t', k,'\t', v.requires_grad)
    #
    #
    # model.add_callback("on_pretrain_routine_start", freeze_model)

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