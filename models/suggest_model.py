from pydantic import BaseModel, Field


class SuggestRequest(BaseModel):
    prefix: str = Field(
        ...,
        min_length=1,
        description="Prefix to autocomplete",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of suggestions",
    )


class SuggestResponse(BaseModel):
    prefix: str
    suggestions: list[str] = []
