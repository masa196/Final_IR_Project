from fastapi import APIRouter, HTTPException

from controllers.refine_controller import refine_query_standalone
from models.refine_model import RefineRequest, RefineResponse


router = APIRouter()


@router.post("/refine", response_model=RefineResponse)
def refine(request: RefineRequest):
    try:
        return refine_query_standalone(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
