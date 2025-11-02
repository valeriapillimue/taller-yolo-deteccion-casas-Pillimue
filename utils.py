import os
import cv2
import matplotlib.pyplot as plt
import shutil
import yaml
from ultralytics import YOLO



def mostrar_imagen_con_boxes(imagen_path, modelo_path, conf=0.25):

    model = YOLO(modelo_path)
    results = model(imagen_path, conf=conf)

    res = results[0]
    img = cv2.imread(imagen_path)

    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Detecciones YOLOv8")
    plt.show()


def contar_archivos_en_dataset(ruta_dataset):

    for subcarpeta in ["train", "test"]:
        img_dir = os.path.join(ruta_dataset, subcarpeta, "images")
        lbl_dir = os.path.join(ruta_dataset, subcarpeta, "labels")

        num_imgs = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
        num_lbls = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0

        print(f"📁 {subcarpeta}: {num_imgs} imágenes, {num_lbls} etiquetas")


def copiar_pesos(origen_best, destino="pesos_guardados"):
    """
    Copia el archivo best.pt a una carpeta local de respaldo.
    """
    os.makedirs(destino, exist_ok=True)
    destino_final = os.path.join(destino, os.path.basename(origen_best))
    shutil.copy(origen_best, destino_final)
    print(f"💾 Pesos copiados en: {destino_final}")
    return destino_final



def leer_data_yaml(ruta_yaml):
    """
    Lee y muestra el contenido de un archivo data.yaml de YOLOv8.
    """
    with open(ruta_yaml, "r") as file:
        data = yaml.safe_load(file)

    print("📄 Información del dataset:")
    for key, value in data.items():
        print(f"   {key}: {value}")
    return data

leer_data_yaml("C:\\Users\\prestamour\\Documents\\taller_yolo_casas\data.yaml")
