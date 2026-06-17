from fastapi import APIRouter

from controllers.health_controller import get_health_status


router = APIRouter()


@router.get("/health")
def health_check():
    return get_health_status()