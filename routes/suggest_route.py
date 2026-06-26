from fastapi import APIRouter, HTTPException

from controllers.suggest_controller import get_suggestions
from models.suggest_model import SuggestRequest, SuggestResponse


router = APIRouter()


@router.post("/suggest", response_model=SuggestResponse)
def suggest(request: SuggestRequest):
    try:
        return get_suggestions(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
