import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurar el middleware para permitir CORS
origins = ["*"]  # Cambia esto según tus necesidades de seguridad
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["*"], allow_headers=["*"])

# Importar y montar el router de predicciones
sys.path.append('./src/app')
import endpoints# import predict_url


app.include_router(endpoints.router, prefix='/zircoins/api/V1')


