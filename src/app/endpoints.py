import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, File, UploadFile

from src.models.predict_model import Predict
from src.features.build_features import ZirconsDataProcessor

# Create an API router
router = APIRouter()

# Load the pre-trained prediction model
predictor = Predict('./models/best_unbalanced_model.pkl')

@router.post("/predict/")
async def predict_url(file: UploadFile = File(...)):
    """
    Endpoint to make predictions based on a CSV or Excel file.

    - **Args:**
        file (UploadFile): File to process and predict.

    - **Returns:**
        JSONResponse: Prediction results in JSON format.
    """
    try:
        # Read the uploaded CSV or Excel file
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file.file)
        else:
            return JSONResponse(content={"error": "Format not supported"}, status_code=400)

        # Process data using the processing class
        processor = ZirconsDataProcessor(data, test=True)
        data = processor.data_processor()

        # Make predictions using the prediction model
        predictions = np.round(predictor.predict(data), 3)

        # Format results as JSON
        predictions_json = jsonable_encoder(predictions.tolist())

        return JSONResponse(content=predictions_json, status_code=200)
    except Exception as e:
        # Handle exceptions and return an error response in case of issues
        return JSONResponse(content={"error": str(e)}, status_code=500)
