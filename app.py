from urllib.parse import quote

import streamlit as st

from rag_model import DEFAULT_TOP_K, generate_answer, load_data

st.set_page_config(page_title="Wikipedia RAG", page_icon="W", layout="wide")

SAMPLE_QUESTIONS = [
    "What is artificial intelligence?",
    "How is machine learning different from deep learning?",
    "What are common uses of AI?",
    "Who is Alan Turing?",
]


def build_wikipedia_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def unique_sources(contexts):
    seen = set()
    sources = []
    for item in contexts:
        source = item["source"]
        if source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return sources


def render_source_links(contexts):
    sources = unique_sources(contexts)
    if not sources:
        return

    st.markdown("**Sources**")
    for source in sources:
        url = build_wikipedia_url(source)
        st.markdown(f"- [{source}]({url})")


def render_source_cards(contexts):
    if not contexts:
        return

    st.markdown("**Sources**")
    seen = set()
    for item in contexts:
        source = item["source"]
        if source in seen:
            continue
        seen.add(source)
        url = build_wikipedia_url(source)
        st.markdown(f"- [{source}]({url})")
        snippet = item.get("snippet", "")
        if snippet:
            st.caption(snippet)


def render_suggestions(suggestions, key_prefix="suggestion"):
    clean_suggestions = []
    seen = set()
    for item in suggestions:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        clean_suggestions.append(normalized)

    if not clean_suggestions:
        return

    st.markdown("**Did you mean:**")
    for index, suggestion in enumerate(clean_suggestions[:3], start=1):
        url = build_wikipedia_url(suggestion)
        st.markdown(f"- [{suggestion}]({url})")
        if st.button(
            f"Ask about {suggestion}",
            key=f"{key_prefix}_{index}_{suggestion}",
            use_container_width=False,
        ):
            st.session_state.prefilled_question = suggestion
            st.rerun()


def render_status_notice(status):
    if status == "empty_query":
        st.info("Please type a question before sending.")
    elif status == "did_you_mean":
        st.warning("I could not confirm an exact Wikipedia match, but I found some close topics below.")
    elif status == "no_match":
        st.warning("I could not find a reliable Wikipedia match for this question.")


def render_confidence(confidence, confidence_score):
    if not confidence:
        return
    st.caption(f"Match quality: {confidence} ({confidence_score:.2f})")


@st.cache_resource
def initialize_rag():
    return load_data()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask a question in plain language. I will look up Wikipedia information "
                "and give you a short, simple answer."
            ),
            "contexts": [],
            "suggestions": [],
            "status": "ready",
            "confidence": "",
            "confidence_score": 0.0,
            "show_details": False,
        }
    ]

st.title("Wikipedia Chat Assistant")
st.caption("Simple answers based on Wikipedia")

try:
    rag = initialize_rag()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Install the dependencies from `requirements.txt`, then make sure the machine "
        "can download Hugging Face models and reach Wikipedia at least once."
    )
    st.stop()

with st.sidebar:
    st.header("Options")
    top_k = st.slider("How much information to search", min_value=1, max_value=5, value=DEFAULT_TOP_K)
    dynamic_search = st.toggle("Search new Wikipedia topics automatically", value=True)
    if st.button("Start a new chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Ask me anything you want to learn about.",
                "contexts": [],
                "suggestions": [],
                "status": "ready",
                "confidence": "",
                "confidence_score": 0.0,
                "show_details": False,
            }
        ]
        st.rerun()

    st.markdown("**Try questions like:**")
    for sample_question in SAMPLE_QUESTIONS:
        st.write(f"- {sample_question}")

    st.markdown(f"Loaded topics: `{len(rag.indexed_topics)}`")
    st.markdown(f"Saved text sections: `{len(rag.chunks)}`")

st.info(
    "Type a question below. You can ask about AI topics already loaded, or leave "
    "`Search new Wikipedia topics automatically` turned on to explore new topics."
)

for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        contexts = message.get("contexts", [])
        suggestions = message.get("suggestions", [])
        status = message.get("status", "answered")
        confidence = message.get("confidence", "")
        confidence_score = message.get("confidence_score", 0.0)
        render_status_notice(status)
        render_confidence(confidence, confidence_score)
        render_source_cards(contexts)
        render_suggestions(suggestions, key_prefix=f"history_{message_index}")
        if contexts and message.get("show_details", False):
            with st.expander("More details", expanded=False):
                for item_number, item in enumerate(contexts, start=1):
                    st.markdown(
                        f"**Source {item_number}: {item['source']}**"
                    )
                    st.write(item["text"])

question_columns = st.columns(len(SAMPLE_QUESTIONS))
for index, sample_question in enumerate(SAMPLE_QUESTIONS):
    if question_columns[index].button(sample_question, use_container_width=True):
        st.session_state.prefilled_question = sample_question

query = st.chat_input("Ask a question about any Wikipedia topic")
if not query and st.session_state.get("prefilled_question"):
    query = st.session_state.pop("prefilled_question")

if query:
    st.session_state.messages.append({"role": "user", "content": query, "contexts": []})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Fetching answers from Wikipedia..."):
            result = generate_answer(
                query,
                top_k=top_k,
                augment_with_search=dynamic_search,
            )
        answer = result.answer
        contexts = result.contexts
        suggestions = result.suggestions
        status = result.status
        confidence = result.confidence
        confidence_score = result.confidence_score
        st.markdown("**Answer**")
        st.markdown(answer)
        render_status_notice(status)
        render_confidence(confidence, confidence_score)
        if contexts:
            render_source_cards(contexts)
            with st.expander("More details", expanded=False):
                for item_number, item in enumerate(contexts, start=1):
                    st.markdown(
                        f"**Source {item_number}: {item['source']}**"
                    )
                    st.write(item["text"])
        elif suggestions:
            render_suggestions(suggestions, key_prefix="live")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "contexts": contexts,
            "suggestions": suggestions,
            "status": status,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "show_details": True,
        }
    )
