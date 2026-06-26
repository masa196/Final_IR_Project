import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from routes.health_route import router as health_router
from routes.search_route import router as search_router
from routes.suggest_route import router as suggest_router
from routes.refine_route import router as refine_router


app = FastAPI(
    title="IR Search Engine API",
    description="API Gateway for the LoTTe IR Search Engine",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "IR Search Engine API Gateway",
        "docs": "/docs",
        "health": "/api/health",
        "search": "/api/search",
    }


app.include_router(
    health_router,
    prefix="/api",
    tags=["Health"],
)

app.include_router(
    search_router,
    prefix="/api",
    tags=["Search"],
)

app.include_router(
    suggest_router,
    prefix="/api",
    tags=["Suggest"],
)

app.include_router(
    refine_router,
    prefix="/api",
    tags=["Refine"],
)