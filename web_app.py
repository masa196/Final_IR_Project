import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000/api"


st.set_page_config(
    page_title="IR Search Engine",
    page_icon="🔎",
    layout="wide",
)


def check_api_health():
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as error:
        return {
            "status": "error",
            "message": str(error),
        }


def search_api(
    query: str,
    model: str,
    top_k: int,
    include_text: bool,
    use_refinement: bool,
    k1: float,
    b: float,
    hybrid_models=None,
    hybrid_fusion_method=None,
    hybrid_weights=None,
    hybrid_first_stage=None,
    hybrid_second_stage=None,
    hybrid_first_stage_k=None,
):
    payload = {
        "query": query,
        "model": model,
        "top_k": top_k,
        "include_text": include_text,
        "use_refinement": use_refinement,
        "k1": k1,
        "b": b,
    }

    if hybrid_models:
        payload["hybrid_models"] = hybrid_models
    if hybrid_fusion_method:
        payload["hybrid_fusion_method"] = hybrid_fusion_method
    if hybrid_weights:
        payload["hybrid_weights"] = hybrid_weights
    if hybrid_first_stage:
        payload["hybrid_first_stage"] = hybrid_first_stage
    if hybrid_second_stage:
        payload["hybrid_second_stage"] = hybrid_second_stage
    if hybrid_first_stage_k:
        payload["hybrid_first_stage_k"] = hybrid_first_stage_k

    response = requests.post(
        f"{API_BASE_URL}/search",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    return response.json()





st.title("🔎 Information Retrieval Search Engine")
st.caption("LoTTe Recreation Search — TF-IDF , BM25 , Embedding and Hybrid Representation")


def suggest_api(prefix: str, top_k: int = 5):
    try:
        response = requests.post(
            f"{API_BASE_URL}/suggest",
            json={"prefix": prefix, "top_k": top_k},
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("suggestions", [])
    except requests.exceptions.RequestException:
        return []





# ---------- Session state initialisation ----------
if "prev_query" not in st.session_state:
    st.session_state.prev_query = ""
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "query_text" not in st.session_state:
    st.session_state.query_text = ""


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Search Settings")

    api_status = check_api_health()

    if api_status.get("status") == "ok":
        st.success("API Gateway is running")
    else:
        st.error("API Gateway is not running")
        st.caption(api_status.get("message"))

    search_mode = st.radio(
        "Search Mode",
        ["Basic Request only", "Basic + Additional Features"],
        index=1,
        help="'Basic + Additional Features' enables spell correction, synonym expansion, history boosting, and autocomplete.",
    )
    is_refined_mode = search_mode == "Basic + Additional Features"

    model_name = st.selectbox(
        "Ranking Model",
        ["BM25", "TF-IDF", "Embedding", "Hybrid Parallel", "Hybrid Serial", "LTR"],
    )

    top_k = st.slider(
        "Number of results",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    include_text = st.checkbox(
        "Show document text",
        value=True,
    )

    is_bm25 = model_name == "BM25"
    is_ltr = model_name == "LTR"
    is_hybrid = model_name in ("Hybrid Parallel", "Hybrid Serial")

    if is_bm25 or is_ltr:
        st.subheader("BM25 / LTR Parameters")
        k1 = st.slider("k1", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        b = st.slider("b", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    elif is_hybrid:
        st.subheader("Hybrid — BM25 Parameters")
        k1 = st.slider("k1", min_value=0.5, max_value=3.0, value=1.5, step=0.1,
                       help="BM25 k1 used when BM25 is included in hybrid")
        b = st.slider("b", min_value=0.0, max_value=1.0, value=0.75, step=0.05,
                      help="BM25 b used when BM25 is included in hybrid")
    else:
        k1 = 1.5
        b = 0.75

    hybrid_models = None
    hybrid_fusion_method = None
    hybrid_weights = None
    hybrid_first_stage = None
    hybrid_second_stage = None
    hybrid_first_stage_k = None



    if model_name == "Hybrid Parallel":
        st.subheader("Hybrid Parallel Settings")
        st.caption("Fixed: BM25 + Embedding with Reciprocal Rank Fusion (RRF)")

        hybrid_models = ["bm25", "embedding"]
        hybrid_fusion_method = "rrf"

    if model_name == "Hybrid Serial":
        st.subheader("Hybrid Serial Settings")
        st.caption("Fixed: BM25 (first stage) → Embedding (second stage)")

        hybrid_first_stage = "bm25"
        hybrid_second_stage = "embedding"

        hybrid_first_stage_k = st.slider(
            "First stage candidates (k)",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Number of candidate documents retrieved in the first stage before re-ranking",
        )



# ---------- Query input (key-bound to session_state) ----------
query = st.text_input(
    "Enter your query",
    placeholder="do bards have to sing?",
    key="query_text",
)

# ---------- Autocomplete suggestion chips ----------
if is_refined_mode and query.strip():
    current_prefix = query.strip().lower()
    if (
        len(current_prefix) >= 2
        and current_prefix != st.session_state.prev_query
    ):
        st.session_state.suggestions = suggest_api(
            current_prefix, top_k=5
        )
        st.session_state.prev_query = current_prefix

    if st.session_state.suggestions:
        st.caption("Suggestions:")
        cols = st.columns(len(st.session_state.suggestions))
        for idx, suggestion in enumerate(st.session_state.suggestions):
            with cols[idx]:
                if st.button(
                    suggestion,
                    key=f"suggest_{idx}",
                    type="tertiary",
                    use_container_width=True,
                ):
                    st.session_state.query_text = suggestion
                    st.rerun()
else:
    st.session_state.suggestions = []


# ---------- Search execution ----------
search_clicked = st.button(
    "Search",
    type="primary",
)


if search_clicked:
    if not query.strip():
        st.warning("Please enter a query.")

    elif api_status.get("status") != "ok":
        st.error(
            "The API Gateway is not running. "
            "Please start FastAPI first."
        )

    else:
        model_for_api = (
            "bm25"
            if model_name == "BM25"
            else "tfidf"
            if model_name == "TF-IDF"
            else "embedding"
            if model_name == "Embedding"
            else "hybrid_parallel"
            if model_name == "Hybrid Parallel"
            else "hybrid_serial"
            if model_name == "Hybrid Serial"
            else "ltr"
        )

        try:
            with st.spinner("Searching through API Gateway..."):
                response = search_api(
                    query=query,
                    model=model_for_api,
                    top_k=top_k,
                    include_text=include_text,
                    use_refinement=is_refined_mode,
                    k1=k1,
                    b=b,
                    hybrid_models=hybrid_models,
                    hybrid_fusion_method=hybrid_fusion_method,
                    hybrid_weights=hybrid_weights,
                    hybrid_first_stage=hybrid_first_stage,
                    hybrid_second_stage=hybrid_second_stage,
                    hybrid_first_stage_k=hybrid_first_stage_k,
                )

            # ---------- Refinement info banner ----------
            if is_refined_mode:
                if response.get("enhanced"):
                    st.success(
                        f"Query Enhanced: "
                        f"**{response['refined_query']}**"
                    )
                else:
                    st.caption("Basic Request — no refinement applied")
            else:
                st.caption("Basic Request — no refinement applied")

            st.subheader("Search Summary")

            is_hybrid_resp = response["model"] in ("hybrid_parallel", "hybrid_serial")

            if is_hybrid_resp:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Model", "HYBRID")
                col2.metric("Type", response.get("hybrid_type", "").upper())
                col3.metric("Top K", response["top_k"])
                col4.metric("Results", len(response["results"]))

                if response.get("fusion_method"):
                    st.info(
                        f"Fusion: {response['fusion_method'].upper()} | "
                        f"Models: {', '.join(response.get('models_used', []))}"
                    )

                if response["model"] == "hybrid_serial":
                    st.info(
                        f"Serial: First stage = {response.get('models_used', ['?'])[0] if response.get('models_used') else '?'} "
                        f"→ Re-ranking stage"
                    )

            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Model", response["model"].upper())
                col2.metric("Top K", response["top_k"])
                col3.metric("Results", len(response["results"]))

            if response["model"] in ("bm25", "ltr") or is_hybrid_resp:
                cols = st.columns(2)
                cols[0].metric("k1", response.get("k1"))
                cols[1].metric("b", response.get("b"))

            if response["model"] != "embedding" or is_hybrid_resp:
                if response.get("processed_query"):
                    st.subheader("Processed Query")
                    st.code(
                        str(response["processed_query"]),
                        language="python",
                    )


            if response["model"] != "embedding":
                st.subheader("Processed Query")
                st.code(
                    str(response["processed_query"]),
                    language="python",
                )



            st.subheader("Results")

            if not response["results"]:
                st.warning("No results found.")

            else:
                for result in response["results"]:
                    with st.container(border=True):
                        st.markdown(
                            f"### Rank {result['rank']}"
                        )

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Doc ID", result["doc_id"])
                        c2.metric("Score", f"{result['score']:.4f}")

                        model_label = response["model"].upper()
                        if is_hybrid_resp:
                            model_label = f"HYBRID ({response.get('hybrid_type', '').upper()})"
                        c3.metric("Model", model_label)

                        if include_text:
                            st.write(
                                result.get(
                                    "text",
                                    "Text was not found.",
                                )
                            )

        except requests.exceptions.HTTPError as error:
            st.error("API returned an error.")
            try:
                st.json(error.response.json())
            except Exception:
                st.write(str(error))

        except requests.exceptions.RequestException as error:
            st.error("Could not connect to the API Gateway.")
            st.write(str(error))
