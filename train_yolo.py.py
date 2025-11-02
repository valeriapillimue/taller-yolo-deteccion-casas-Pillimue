
from roboflow import Roboflow
from ultralytics import YOLO
import os


def descargar_dataset(api_key: str, workspace: str, proyecto: str, version_num: int, formato: str = "yolov8"):

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(proyecto)
    version = project.version(version_num)
    dataset = version.download(formato)
    print(f"✅ Dataset descargado correctamente en: {dataset.location}")
    return dataset.location

def entrenar_modelo(ruta_pesos_base: str, ruta_data_yaml: str, epochs: int = 50, imgsz: int = 640,
                    paciencia: int = 20, nombre_proyecto: str = "experimentos", nombre_run: str = "v1",
                    conf_cls: float = 0.8, usar_rect: bool = True, guardar_plots: bool = True, validar: bool = False):
 
    model = YOLO(ruta_pesos_base)

    print("🚀 Iniciando entrenamiento de YOLOv8...")
    results = model.train(
        data=ruta_data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        patience=paciencia,
        project=nombre_proyecto,
        name=nombre_run,
        cls=conf_cls,
        rect=usar_rect,
        plots=guardar_plots,
        val=validar
    )
    print("✅ Entrenamiento completado.")
    return model



