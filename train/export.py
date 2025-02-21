from ultralytics import YOLO
from ultralytics import RTDETR


model_paths = [
    r'D:\Projects\20240701_C220806_东莞思泉石墨片卷料Copy\weights\20240702\best.pt',
    # r'T:\train_data_zzg\Code\bohr_yolov8\runs\segment\思泉_石墨卷_copy_2024061518092\weights\b_optimizer.pt',
    # r'E:\bohr_projects\JGP-4x&9x\B403\pt\AQR.pt',
    # r'E:\bohr_projects\JGP-4x&9x\B403\pt\CR.pt',
    # r'E:\bohr_projects\JGP-4x&9x\B403\pt\D.pt',
    # r'E:\bohr_projects\JGP-4x&9x\B403\pt\S.pt',
]

for model_path in model_paths:
    # Load a model
    model = YOLO(model_path)
    # Export the model
    model.export(format='onnx', simplify=True, imgsz=[1280, 1280], batch=1)

    print(f"model_path :{model_path}, 已完成！")
