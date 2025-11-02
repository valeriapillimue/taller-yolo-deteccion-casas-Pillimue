# Taller Técnico: Detector de Casas con YOLOv8


## Objetivo:

Este repositorio contiene la implementación completa de un flujo de trabajo de Detección de Objetos, siguiendo el Taller de Visión Computacional.El objetivo principal fue crear una base de datos con casas, anotar el dataset en formato YOLO y entrenar un modelo YOLOv8-n para localizar casas  en escenas.

## Dataset

Se tomaron imágenes de diferentes casas colombianas desde google maps y, del dataset https://images.cv/dataset/house-image-classification-dataset.
Se Obtuvieron 80 imágenes, sin embargo, luego de técticas de aumento de datos: 
* Auto-orientación
* Salidas por ejemplo de entrenamiento: 3
* Voltear: Horizontal
* Rotación: Entre -3° y +3°
* Saturación: Entre -34% y +34%

* Desenfoque: Hasta 1.2px
Se logró una cantidad de 208 imágenes. La división del dataset 192 imágenes en train y 16 de entrenamiento.
Anotaciones, bounding boxes, aumentación de datos con la herramienta Roboflow.

## Instrucciones de Reproducción 

### Requisitos

Asegúrate de tener Python 3.x instalado. En requirements.txt se puede visualizar qué librerias son necesarias.

**Instalar dependencias:**
    ```bash
    pip install ultralytics fastapi uvicorn python-multipart pillow
    pip install roboflow
    pip install -U numpy==1.26.4
    pip install -U ultralytics torch torchvision
    ```

### Entrenamiento

Para reproducir el entrenamiento, con el archivo train_yolo.py se requiere de acceso por clave a Roboflow, así que con la función descargar_dataset() se puede obetener el dataset y con la función entrenar_modelo se puede entrenar con diferentes configuraciones, mis configuraciones:

```bash
model = YOLO("yolov8m.pt")
results = model.train(
    data="/content/CASAS-2/data.yaml",
    epochs=50,
    imgsz=640,
    patience=20,
    project ='casas_exp',
    name = 'v2',
    cls = 0.8,
    rect = True,
    plots = True,
    val=False
)
```

### Inferencias

Con el archivo inferencia.py puede observar la lógica para cargar el modelo entrenado, ejecutar la detección sobre una imagen y visualizar los resultados con las cajas delimitadoras y los puntajes de confianza.

Con la función carga_pesos() se carga el modelo YOLO desde el ruta de los pesos guardados (.pt)
Para ejecutar esta función solo es necesario la ruta de los pesos.

Con la función ejecutar_inferencia() se ejecuta la predicción. Extrae las cajas (xyxy), las confianzas (conf) y las clases (cls). Dibuja las cajas y la etiqueta de confianza sobre la imagen usando OpenCV (cv2) y guarda el resultado. Además, muestra la imagen usando Matplotlib si mostrar=True.
Para ejecutar esta función solo es necesario el modelo y la ruta de una imágen. 


### Utilidades

El archivo utils.py contiene varias funciones auxiliares diseñadas para simplificar la gestión del dataset, la configuración y la validación visual del modelo fuera del proceso principal de entrenamiento e inferencia.

**leer_data_yaml:**Lee el archivo data.yaml y muestra en consola la configuración clave del dataset (paths, número de clases, nombres de clase). Es útil para verificar la configuración antes de entrenar.
**contar_archivos_en_dataset:**Realiza una verificación del dataset contando el número de archivos de imagen y de etiquetas (.txt) presentes en las carpetas train y test/val. Ayuda a confirmar que el split de datos se hizo correctamente.
**mostrar_imagen_con_boxes:**Herramienta de validación visual rápida. Carga un modelo y una imagen, ejecuta la inferencia y muestra la imagen con las cajas delimitadoras, las etiquetas de clase y el puntaje de confianza (score) dibujados encima.
**copiar_pesos**Función de gestión de archivos que copia los pesos entrenados (best.pt) a una carpeta de respaldo específica (models/ u otra definida), asegurando que los pesos finales estén disponibles para el despliegue.

### Despliegue de la API REST

Este endpoint permite a cualquier cliente enviar una imagen y recibir las detecciones de casas en un formato JSON estandarizado, que incluye el puntaje de confianza (score) y las coordenadas de las cajas (bbox). La API devuelve una lista de objetos JSON que cumplen con el esquema de Pydantic DetectionResult, garantizando que la salida incluye las cajas y los scores solicitados. El código del servidor se encuentra en API.py.

**Ejecutar el servidor:** Desde el directorio principal del proyecto, ejecute el siguiente comando.
uvicorn API:app --reload

* El servidor se iniciará y estará disponible en http://127.0.0.1:8000.

* El endpoint se puede probar y documentar automáticamente accediendo a la documentación interactiva de FastAPI: http://127.0.0.1:8000/docs.

## Resultados y ejemplos de detección

Para las métricas se decidió: recall, matriz de confusión, precisión.

Revisaremos la matriz de confusión
<img width="3000" height="2250" alt="matrix-confusion" src="https://github.com/user-attachments/assets/b8dc8181-014a-48d1-aba9-2410cb31b264" />
Vemos una buena cantidad de de verdaderos positivos pero no detecta bien todas las casas

La métrica mAP@0.5 de 0.681 indica que, en promedio, el modelo logra identificar las casas con una Intersection Over Union (IOU) mayor al 50% en un 68.1% de los casos. 

El modelo exhibe una Precision de 0.788, lo que significa que el 78.8% de las detecciones que el modelo clasificó como "casa" son correctas. Esta es una buena cifra y sugiere que el modelo no produce una cantidad excesiva de Falsos Positivos. En otras palabras, cuando el modelo predice una casa, es bastante fiable.

Sin embargo, el Recall (Exhaustividad) de 0.620 es relativamente bajo. Un recall del 62.0% indica que el modelo solo está encontrando cerca de dos tercios de todas las casas que realmente existen en el conjunto de validación. Este valor sugiere un problema de Falsos Negativos, donde el modelo está fallando en detectar casas que son visibles en la imagen.

La disparidad entre la alta Precision y el bajo Recall es el punto clave del análisis: el modelo es selectivo (preciso), pero no es exhaustivo (tiene fallas al encontrar todas las casas). Dado que el dataset utilizado es pequeño, esta limitación es esperable. La falta de variabilidad en ángulos, tamaños y condiciones de iluminación en las imágenes recolectadas provoca que el modelo tenga un buen desempeño en los casos que aprendió bien, pero falle en generalizar. Otro factor que afectó es que el dataset se conforma por casas colombianas y casas de otros paises, al ser casas con estéticas diferentes y tan pocos datos se afecta el entrenamiento.

![Ejemplo de una correcta y alta detección de casa](https://github.com/user-attachments/assets/a8c3b0d1-2155-4cf7-9a07-86dd72ed78ec)

![Obtiene mucho fondo en la caja](https://github.com/user-attachments/assets/95ea6703-2405-4338-ab5f-ec67cef567eb)

![No logra capturar todas las casas y los pesos de estas casas son pequeños](https://github.com/user-attachments/assets/7e6bebb3-cf2f-458b-8b92-528e9f487d6e)


## Limitaciones y pasos futuros

Al ser un trabajo académico y breve, se trabajó poco en la recolección de datos y el etiquetado manual, las cosas pueden cambiar mucho por la recolección de datos. Se debe aumentar el dataset ya que el modelo es susceptible al sobreajuste (overfitting).

Mejorar la robustez del endpoint de la API con manejo de errores de imagen más detallado
