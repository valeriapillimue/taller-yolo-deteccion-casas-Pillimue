from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

def cargar_pesos(ruta_pesos: str):

    model = YOLO(ruta_pesos)
    print(f"✅ Modelo cargado correctamente desde: {ruta_pesos}")
    return model


def ejecutar_inferencia(modelo, ruta_imagen: str, carpeta_salida: str = "resultados", mostrar: bool = True):

    os.makedirs(carpeta_salida, exist_ok=True)
    resultados = modelo(ruta_imagen)

    res = resultados[0]
    cajas = res.boxes.xyxy.cpu().numpy()  
    confs = res.boxes.conf.cpu().numpy()  
    clases = res.boxes.cls.cpu().numpy() 

    imagen = cv2.imread(ruta_imagen)
    for i, box in enumerate(cajas):
        x1, y1, x2, y2 = map(int, box)
        conf = confs[i]
        cls = int(clases[i])
        etiqueta = f"{modelo.names[cls]}: {conf:.2f}"

        # Dibujar rectángulo y texto
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(imagen, etiqueta, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    nombre_salida = os.path.join(carpeta_salida, os.path.basename(ruta_imagen))
    cv2.imwrite(nombre_salida, imagen)
    print(f"📁 Imagen con detecciones guardada en: {nombre_salida}")

    #Mostrar resultados en pantalla
    if mostrar:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Detecciones YOLOv8")
        plt.show()

    # Imprimir resultados resumidos
    print("\n🔍 Resultados de detección:")
    for i, box in enumerate(cajas):
        print(f"Objeto {i+1}: {modelo.names[int(clases[i])]} "
              f"Confianza: {confs[i]:.2f} - Coordenadas: {box}")
    


ruta_pesos = "taller_yolo_casas\\casas_model_best.pt"           # ruta a tu modelo entrenado
ruta_imagen = "taller_yolo_casas\\Captura de pantalla 2025-11-01 102020.png" # ruta a una imagen para probar

modelo = cargar_pesos(ruta_pesos)
ejecutar_inferencia(modelo, ruta_imagen, mostrar=True)
