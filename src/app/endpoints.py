import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, File, UploadFile

from src.models.predict_model import Predict
from src.features.build_features import ZirconsDataProcessor



router = APIRouter()

predictor = Predict('./models/best_unbalanced_model.pkl')


@router.post("/predict/")
async def predict_url(file: UploadFile = File(...)):
    try:
        # Leer el archivo CSV o Excel enviado
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file.file)
        else:
            return JSONResponse(content={"error": "Format not supported"}, status_code=400)


        processor = ZirconsDataProcessor(data, test =True)
        data = processor.data_processor()
        # Realizar predicciones
        predictions = np.round(predictor.predict(data),3)

        # Formatear resultados como JSON
        predictions_json = jsonable_encoder(predictions.tolist())

        return JSONResponse(content=predictions_json, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
