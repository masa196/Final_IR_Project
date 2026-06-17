from fastapi import APIRouter, HTTPException

from models.search_model import (
    SearchRequest,
    SearchResponse,
)

from controllers.search_controller import search_documents


router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
)
def search(request: SearchRequest):
    try:
        return search_documents(request)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {error}",
        )