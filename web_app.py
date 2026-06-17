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
    k1: float,
    b: float,
):
    payload = {
        "query": query,
        "model": model,
        "top_k": top_k,
        "include_text": include_text,
        "k1": k1,
        "b": b,
    }

    response = requests.post(
        f"{API_BASE_URL}/search",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


st.title("🔎 Information Retrieval Search Engine")
st.caption("LoTTe Recreation Search — TF-IDF , BM25 and Embedding ")


with st.sidebar:
    st.header("Search Settings")

    api_status = check_api_health()

    if api_status.get("status") == "ok":
        st.success("API Gateway is running")
    else:
        st.error("API Gateway is not running")
        st.caption(api_status.get("message"))

    model_name = st.selectbox(
        "Ranking Model",
        ["BM25", "TF-IDF", "Embedding"],
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

    if model_name == "BM25":
        st.subheader("BM25 Parameters")

        k1 = st.slider(
            "k1",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
        )

        b = st.slider(
            "b",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
        )

    else:
        k1 = 1.5
        b = 0.75


query = st.text_input(
    "Enter your query",
    placeholder="do bards have to sing?",
)


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
        )

        try:
            with st.spinner("Searching through API Gateway..."):
                response = search_api(
                    query=query,
                    model=model_for_api,
                    top_k=top_k,
                    include_text=include_text,
                    k1=k1,
                    b=b,
                )

            st.subheader("Search Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Model",
                response["model"].upper(),
            )

            col2.metric(
                "Top K",
                response["top_k"],
            )

            col3.metric(
                "Results",
                len(response["results"]),
            )

            if response["model"] == "bm25":
                col4, col5 = st.columns(2)

                col4.metric(
                    "k1",
                    response.get("k1"),
                )

                col5.metric(
                    "b",
                    response.get("b"),
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

                        c1.metric(
                            "Doc ID",
                            result["doc_id"],
                        )

                        c2.metric(
                            "Score",
                            f"{result['score']:.4f}",
                        )

                        c3.metric(
                            "Model",
                            response["model"].upper(),
                        )

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