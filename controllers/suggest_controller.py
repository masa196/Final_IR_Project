import logging

from models.suggest_model import SuggestRequest, SuggestResponse
from services.query_refinement.trie_service import get_vocabulary_trie

logger = logging.getLogger(__name__)


def get_suggestions(request: SuggestRequest) -> SuggestResponse:
    try:
        trie = get_vocabulary_trie()
        suggestions = trie.suggest(request.prefix.lower(), limit=request.top_k)
        return SuggestResponse(prefix=request.prefix, suggestions=suggestions)
    except Exception as exc:
        logger.warning("Autocomplete suggestion failed: %s", exc)
        return SuggestResponse(prefix=request.prefix, suggestions=[])
