def get_health_status():
    return {
        "status": "ok",
        "message": "IR Search API is running",
        "available_models": [
            "tfidf",
            "bm25",
            "embedding",
            "ltr",
        ],
    }