import logging

from models.refine_model import RefineRequest, RefineResponse
from services.query_refinement.query_refinement_service import (
    correct_spelling,
    expand_synonyms,
    boost_from_history,
    SearchHistory,
)
from services.preprocessing.preprocessing_service import preprocess_lotte

logger = logging.getLogger(__name__)


def refine_query_standalone(request: RefineRequest) -> RefineResponse:
    try:
        corrected = correct_spelling(request.query)
        tokens = preprocess_lotte(corrected)
        expanded = expand_synonyms(tokens)

        if request.use_history:
            boosted = boost_from_history(expanded)
            SearchHistory.record_query(expanded)
        else:
            boosted = list(expanded)

        refined = " ".join(boosted)
        enhanced = refined != request.query

        return RefineResponse(
            original_query=request.query,
            refined_query=refined,
            corrected=corrected,
            expanded_tokens=expanded,
            boosted_tokens=boosted,
            enhanced=enhanced,
        )
    except Exception as exc:
        logger.warning("Query refinement failed: %s", exc)
        return RefineResponse(
            original_query=request.query,
            refined_query=request.query,
            corrected=request.query,
            expanded_tokens=[],
            boosted_tokens=[],
            enhanced=False,
        )
