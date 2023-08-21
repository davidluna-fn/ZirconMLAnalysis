import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create a FastAPI app instance
app = FastAPI()

# Configure CORS middleware
origins = ["*"]  # Update this list based on your security needs
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["*"], allow_headers=["*"])

# Import and mount the prediction router
from src.app import endpoints

# Include the prediction router under the specified prefix
app.include_router(endpoints.router, prefix='/zircoins/api/V1')
