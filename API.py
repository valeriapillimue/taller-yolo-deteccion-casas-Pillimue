import io
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
from ultralytics import YOLO

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionResult(BaseModel):
    class_name: str
    score: float
    bbox: BoundingBox

app = FastAPI(title="YOLO House Detector API")

MODEL_PATH = "RUTA\\taller_yolo_casas\\casas_model_best.pt"

model = YOLO(MODEL_PATH)

#Endpoint para la detección de objetos
@app.post("/detect", response_model=list[DetectionResult], summary="Ejecuta la detección de casas en una imagen")

async def detect_houses(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()
        
        #Convertir bytes a objeto de imagen PIL
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        #Convertir a array de NumPy (formato preferido por YOLO)
        image_np = np.array(image_pil)

        results = model(image_np, verbose=False, stream=False)
        detections = []
        for result in results:
            boxes = result.boxes            
            for box in boxes:
                class_id = int(box.cls[0].item())
                score = round(box.conf[0].item(), 3)                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = model.names[class_id] 
                detection = DetectionResult(
                    class_name=class_name,
                    score=score,
                    bbox=BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2
                    )
                )
                detections.append(detection)

        return detections

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error durante la detección: {e}"})
'''
# Instrucción para ejecutar el servidor 
if __name__ == "__main__":
    # Comando para iniciar el servidor en el puerto 8000
    # Accede a la documentación interactiva en http://127.0.0.1:8000/docs
    uvicorn.run("API:app", host="0.0.0.0", port=8000, reload=True)
'''