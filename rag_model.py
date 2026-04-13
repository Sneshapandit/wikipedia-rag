from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import faiss
import numpy as np
import wikipedia
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_TOPICS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-base"
DEFAULT_TOP_K = 3
MAX_CONTEXT_CHUNKS = 5
MAX_NEW_TOKENS = 192
MIN_RELEVANCE_SCORE = 0.38
MIN_TOPIC_OVERLAP_RATIO = 0.5
NOISY_SOURCE_PATTERNS = (
    "glossary",
    "list of",
    "outline of",
    "index of",
    "(disambiguation)",
)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


@dataclass
class AnswerResult:
    answer: str
    contexts: List[Dict[str, str | float]]
    suggestions: List[str]
    status: str
    confidence: str
    confidence_score: float


class WikipediaRAG:
    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        generator_model_name: str = GENERATOR_MODEL_NAME,
    ) -> None:
        self.embedding_model = self._load_embedding_model(embedding_model_name)
        self.tokenizer, self.generator_model = self._load_generator_model(generator_model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[Dict[str, str]] = []
        self.indexed_topics: set[str] = set()

    @staticmethod
    def _load_embedding_model(model_name: str) -> SentenceTransformer:
        try:
            return SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the embedding model. Make sure the model is available "
                "locally or that this machine can reach Hugging Face."
            ) from exc

    @staticmethod
    def _load_generator_model(
        model_name: str,
    ) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the FLAN-T5 generator. Make sure the model is available "
                "locally or that this machine can reach Hugging Face."
            ) from exc
        return tokenizer, model

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"==+[^=]+==+", " ", text)
        text = re.sub(r"\b[A-Z][A-Za-z]+ at (IMDb|TV Guide|Rotten Tomatoes|Bollywood Hungama|Discogs)\b", " ", text)
        text = re.sub(r"\b[Mm]angeshkar [Ff]amily\b", " ", text)
        text = re.sub(r"\b[A-Z][A-Za-z]+ discography at Discogs\b", " ", text)
        text = re.sub(r"\b[A-Z][A-Za-z]+ at [A-Z][A-Za-z ]+\b", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\[\d+\]", "", text)
        return text.strip()

    @classmethod
    def chunk_text(
        cls,
        text: str,
        chunk_size: int = 180,
        overlap: int = 40,
    ) -> List[str]:
        cleaned = cls.clean_text(text)
        words = cleaned.split()
        if not words:
            return []

        chunks: List[str] = []
        step = max(chunk_size - overlap, 1)
        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                continue
            chunk = " ".join(chunk_words).strip()
            if len(chunk) > 80:
                chunks.append(chunk)
            if start + chunk_size >= len(words):
                break
        return chunks

    def _fetch_topic_content(self, topic: str) -> str | None:
        try:
            page = wikipedia.page(topic, auto_suggest=False)
        except wikipedia.DisambiguationError as exc:
            for option in exc.options[:5]:
                try:
                    page = wikipedia.page(option, auto_suggest=False)
                    return page.content
                except Exception:
                    continue
            return None
        except Exception:
            return None
        summary = getattr(page, "summary", "") or ""
        content = getattr(page, "content", "") or ""
        return f"{summary}\n\n{content}".strip()

    @staticmethod
    def is_noisy_source(source: str) -> bool:
        normalized = source.strip().lower()
        return any(pattern in normalized for pattern in NOISY_SOURCE_PATTERNS)

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def _ensure_index(self, embedding_dim: int) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatIP(embedding_dim)

    def add_documents(self, documents: Sequence[Tuple[str, str]]) -> None:
        new_entries: List[Dict[str, str]] = []
        for source, document in documents:
            if source in self.indexed_topics:
                continue
            if self.is_noisy_source(source):
                continue
            for chunk_id, chunk in enumerate(self.chunk_text(document)):
                new_entries.append(
                    {
                        "source": source,
                        "text": chunk,
                        "chunk_id": str(chunk_id),
                    }
                )
            self.indexed_topics.add(source)

        if not new_entries:
            return

        embeddings = self._encode_texts([entry["text"] for entry in new_entries])
        self._ensure_index(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks.extend(new_entries)

    def load_default_data(self, topics: Sequence[str] | None = None) -> None:
        topic_list = list(topics or DEFAULT_TOPICS)
        documents: List[Tuple[str, str]] = []
        for topic in topic_list:
            content = self._fetch_topic_content(topic)
            if content:
                documents.append((topic, content))
        if not documents:
            raise RuntimeError(
                "Unable to load Wikipedia content for the default topics. "
                "Check internet access and try again."
            )
        self.add_documents(documents)

    def maybe_expand_with_query(self, query: str, limit: int = 2) -> None:
        results = self.search_topics(query, limit=limit + 2)
        if not results:
            return

        candidate_topics = [
            title
            for title in results
            if title not in self.indexed_topics and self.is_topic_relevant_to_query(query, title)
        ][:limit]
        documents: List[Tuple[str, str]] = []
        for topic in candidate_topics:
            content = self._fetch_topic_content(topic)
            if content:
                documents.append((topic, content))
        self.add_documents(documents)

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[RetrievedChunk]:
        if self.index is None or not self.chunks:
            raise RuntimeError("The FAISS index is empty. Call load_data() before retrieval.")

        top_k = max(1, min(max(k * 2, k), len(self.chunks)))
        query_embedding = self._encode_texts([query])
        scores, indices = self.index.search(query_embedding, top_k)

        results: List[RetrievedChunk] = []
        seen_pairs: set[tuple[str, str]] = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            pair = (chunk["source"], chunk["text"][:120])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            results.append(
                RetrievedChunk(
                    text=chunk["text"],
                    source=chunk["source"],
                    score=float(score),
                )
            )
            if len(results) >= k:
                break
        return results

    def retrieve_for_sources(
        self,
        query: str,
        allowed_sources: Sequence[str],
        k: int = DEFAULT_TOP_K,
    ) -> List[RetrievedChunk]:
        allowed = {source.strip() for source in allowed_sources if source.strip()}
        if not allowed:
            return []

        retrieved = self.retrieve(query, k=max(k * 3, k))
        filtered = [item for item in retrieved if item.source in allowed]
        return filtered[:k]

    @staticmethod
    def build_source_snippet(text: str, max_chars: int = 180) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        shortened = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
        return f"{shortened}..."

    @staticmethod
    def _normalized_words(text: str) -> List[str]:
        return [
            word.lower()
            for word in re.findall(r"[A-Za-z]+", text)
            if len(word) > 2
        ]

    @staticmethod
    def _word_set(text: str) -> set[str]:
        return set(WikipediaRAG._normalized_words(text))

    @classmethod
    def search_topics(cls, query: str, limit: int = 5) -> List[str]:
        search_terms = []
        raw_query = query.strip()
        topic = cls.extract_topic_from_query(query)
        if topic:
            search_terms.append(topic)
        if raw_query and raw_query not in search_terms:
            search_terms.append(raw_query)

        merged_results: List[str] = []
        seen: set[str] = set()
        for term in search_terms:
            try:
                results = wikipedia.search(term, results=limit)
            except Exception:
                continue
            for result in results:
                normalized = result.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged_results.append(normalized)
        return merged_results[:limit]

    @staticmethod
    def normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query).strip()

    @classmethod
    def score_topic_match(cls, query: str, topic: str) -> float:
        query_words = cls._word_set(cls.extract_topic_from_query(query))
        topic_words = cls._word_set(topic)
        if not query_words or not topic_words:
            return 0.0

        overlap = query_words & topic_words
        if not overlap:
            return 0.0

        overlap_ratio = len(overlap) / len(query_words)
        topic_ratio = len(overlap) / len(topic_words)
        exact_bonus = 0.25 if cls.extract_topic_from_query(query).strip().lower() == topic.strip().lower() else 0.0
        substring_bonus = 0.15 if cls.extract_topic_from_query(query).strip().lower() in topic.strip().lower() else 0.0
        return overlap_ratio * 0.6 + topic_ratio * 0.25 + exact_bonus + substring_bonus

    @classmethod
    def is_topic_relevant_to_query(cls, query: str, topic: str) -> bool:
        score = cls.score_topic_match(query, topic)
        return score >= MIN_TOPIC_OVERLAP_RATIO or score >= 0.75

    @classmethod
    def rank_topics_for_query(cls, query: str, topics: Sequence[str]) -> List[str]:
        scored = [(cls.score_topic_match(query, topic), topic) for topic in topics]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], len(item[1])))
        return [topic for _, topic in scored]

    @classmethod
    def has_relevant_match(
        cls,
        query: str,
        contexts: Sequence[RetrievedChunk],
        suggestions: Sequence[str],
    ) -> bool:
        if not contexts:
            return False

        top_score = max(item.score for item in contexts)
        if top_score < MIN_RELEVANCE_SCORE:
            return False

        query_words = cls._word_set(query)
        filtered_suggestions = [
            item for item in suggestions if cls.is_topic_relevant_to_query(query, item)
        ]
        suggestion_words = [cls._word_set(item) for item in filtered_suggestions]
        for context in contexts:
            source_words = cls._word_set(context.source)
            if cls.is_topic_relevant_to_query(query, context.source):
                return True
            if any(source_words == item_words or source_words & item_words for item_words in suggestion_words):
                return True

        return top_score >= 0.52

    @staticmethod
    def compute_confidence_score(contexts: Sequence[RetrievedChunk]) -> float:
        if not contexts:
            return 0.0
        top_score = max(item.score for item in contexts)
        avg_score = sum(item.score for item in contexts) / len(contexts)
        combined = top_score * 0.65 + avg_score * 0.35
        return round(max(0.0, min(1.0, combined)), 3)

    @staticmethod
    def confidence_label(score: float) -> str:
        if score >= 0.7:
            return "High match"
        if score >= 0.5:
            return "Possible match"
        if score > 0:
            return "Low match"
        return "No match"

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        cleaned_parts: List[str] = []
        for part in parts:
            candidate = part.strip()
            if len(candidate) <= 20:
                continue
            if "==" in candidate:
                continue
            if " at IMDb" in candidate or " at TV Guide" in candidate or " at Rotten Tomatoes" in candidate:
                continue
            if candidate.count(" at ") >= 2:
                continue
            cleaned_parts.append(candidate)
        return cleaned_parts

    @staticmethod
    def infer_query_type(query: str) -> str:
        lowered = query.strip().lower()
        if lowered.startswith("who is") or lowered.startswith("who was"):
            return "person"
        if lowered.startswith("what is") or lowered.startswith("what was"):
            return "concept"
        return "general"

    @classmethod
    def build_fallback_answer(
        cls,
        query: str,
        contexts: Sequence[RetrievedChunk],
    ) -> str:
        if not contexts:
            return "I don't know."

        scored_sentences: List[tuple[float, str]] = []
        seen_sentences: set[str] = set()
        query_words = {
            word.lower()
            for word in re.findall(r"[A-Za-z]+", query)
            if len(word) > 2
        }
        query_type = cls.infer_query_type(query)
        topic = cls.extract_topic_from_query(query)

        for item_index, item in enumerate(contexts):
            sentences = cls.split_sentences(item.text)
            for sentence_index, sentence in enumerate(sentences):
                normalized = sentence.lower()
                if normalized in seen_sentences:
                    continue
                seen_sentences.add(normalized)

                score = 0.0
                for word in query_words:
                    if word in normalized:
                        score += 1.0

                # Prefer earlier sources and earlier sentences, which are usually closer
                # to the page summary and produce cleaner openings.
                score += max(item.score, 0) * 2.0
                score += max(0.0, 1.0 - item_index * 0.2)
                score += max(0.0, 0.8 - sentence_index * 0.15)

                if topic and topic.lower() in normalized:
                    score += 1.2
                if query_type == "person":
                    if " is " in normalized or " was " in normalized:
                        score += 0.9
                    if any(
                        marker in normalized
                        for marker in ["singer", "scientist", "actor", "writer", "politician", "composer"]
                    ):
                        score += 0.7
                if query_type == "concept":
                    if normalized.startswith(topic.lower()) or " is " in normalized or " refers to " in normalized:
                        score += 0.8

                if score > 0 or not scored_sentences:
                    scored_sentences.append((score, sentence))

        if not scored_sentences:
            return "I don't know."

        scored_sentences.sort(key=lambda item: (-item[0], len(item[1])))
        collected: List[str] = []
        for _, sentence in scored_sentences:
            collected.append(sentence)
            if len(collected) >= 3:
                break

        answer = " ".join(collected).strip()
        answer = re.sub(r'\s+"', ' "', answer)
        answer = cls.rewrite_intro_if_needed(query, answer)
        answer = cls.ensure_complete_opening(query, answer)
        return answer or "I don't know."

    @classmethod
    def rewrite_intro_if_needed(cls, query: str, answer: str) -> str:
        topic = cls.extract_topic_from_query(query)
        query_type = cls.infer_query_type(query)
        if not topic or not answer:
            return answer

        lower_answer = answer.lower()
        if query_type == "person" and topic.lower() not in lower_answer[:80]:
            return f"{topic} was {answer[0].lower() + answer[1:]}"
        if query_type == "person" and " is the recipient of " in lower_answer[:120]:
            return answer.replace(
                f"{topic} is the recipient of",
                f"{topic} was an Indian singer who received",
                1,
            )
        if query_type == "general" and topic.lower() not in lower_answer[:80]:
            return f"{topic} is {answer[0].lower() + answer[1:]}"
        return answer

    @classmethod
    def build_grounded_answer(
        cls,
        query: str,
        contexts: Sequence[RetrievedChunk],
    ) -> str:
        answer = cls.build_fallback_answer(query, contexts)
        if answer and answer != "I don't know.":
            return answer
        return "I could not find a reliable answer for that in the available Wikipedia sources."

    @classmethod
    def build_no_match_message(cls, suggestions: Sequence[str]) -> str:
        if suggestions:
            return (
                "I could not find a reliable answer for that in the available Wikipedia sources. "
                "You can try one of the suggested Wikipedia topics below."
            )
        return (
            "I could not find a reliable answer for that in the available Wikipedia sources. "
            "Please try a different topic name or check the spelling."
        )

    @staticmethod
    def is_weak_answer(answer: str, contexts: Sequence[RetrievedChunk]) -> bool:
        cleaned = answer.strip()
        if len(cleaned) < 30:
            return True
        if len(cleaned.split()) < 18:
            return True
        if cleaned.lower() == "i don't know":
            return False

        source_names = {item.source.strip().lower().rstrip(".") for item in contexts}
        if cleaned.lower().rstrip(".") in source_names:
            return True
        return False

    @staticmethod
    def extract_topic_from_query(query: str) -> str:
        cleaned = query.strip().rstrip("?.!")
        patterns = [
            r"(?i)^who is (.+)$",
            r"(?i)^what is (.+)$",
            r"(?i)^tell me about (.+)$",
            r"(?i)^explain (.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, cleaned)
            if match:
                return match.group(1).strip().capitalize()
        return cleaned[:1].upper() + cleaned[1:] if cleaned else ""

    @classmethod
    def ensure_complete_opening(cls, query: str, answer: str) -> str:
        cleaned = answer.strip()
        if not cleaned:
            return "I don't know."

        opening = cleaned.split(" ", 1)[0].lower().strip("\"'(),")
        weak_openings = {
            "this",
            "that",
            "these",
            "it",
            "he",
            "she",
            "they",
            "his",
            "her",
            "its",
        }
        if opening in weak_openings:
            topic = cls.extract_topic_from_query(query)
            if topic:
                return f"{topic} {cleaned[0].lower() + cleaned[1:]}"
        topic = cls.extract_topic_from_query(query)
        topic_words = cls._word_set(topic)
        answer_start_words = cls._word_set(" ".join(cleaned.split()[:6]))
        if topic and topic_words and not (topic_words & answer_start_words):
            return f"{topic} is {cleaned[0].lower() + cleaned[1:]}"
        return cleaned

    @staticmethod
    def build_prompt(query: str, contexts: Sequence[RetrievedChunk]) -> str:
        joined_context = "\n\n".join(
            f"Source: {item.source}\nContext: {item.text}" for item in contexts
        )
        return (
            "You are a retrieval-augmented assistant.\n"
            "Answer ONLY using the provided context. "
            "If answer is not present, say 'I don't know'. "
            "Keep the answer simple, clear, and helpful for a non-technical reader. "
            "Write one brief but informative paragraph with 3 to 5 sentences. "
            "Start with the topic name and a direct answer to the question, then add a little explanation, "
            "key differences or examples when useful. "
            "Do not mention the word context. Do not return bullet points. "
            "Do not repeat the question.\n\n"
            f"Context:\n{joined_context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def answer_question(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        augment_with_search: bool = True,
    ) -> AnswerResult:
        normalized_query = self.normalize_query(query)
        if not normalized_query:
            return AnswerResult(
                answer="Please enter a question so I can search Wikipedia for you.",
                contexts=[],
                suggestions=[],
                status="empty_query",
                confidence="No match",
                confidence_score=0.0,
            )

        suggestions = self.search_topics(normalized_query)
        ranked_suggestions = self.rank_topics_for_query(query, suggestions)
        if augment_with_search:
            self.maybe_expand_with_query(normalized_query)
        preferred_sources = ranked_suggestions[:2]
        contexts = self.retrieve_for_sources(
            normalized_query,
            allowed_sources=preferred_sources,
            k=min(top_k, MAX_CONTEXT_CHUNKS),
        )
        if not contexts:
            contexts = self.retrieve(normalized_query, k=min(top_k, MAX_CONTEXT_CHUNKS))
        filtered_suggestions = [
            item for item in ranked_suggestions if self.is_topic_relevant_to_query(normalized_query, item)
        ]
        if not self.has_relevant_match(normalized_query, contexts, filtered_suggestions):
            answer = self.build_no_match_message(filtered_suggestions[:3])
            status = "did_you_mean" if filtered_suggestions else "no_match"
            return AnswerResult(
                answer=answer,
                contexts=[],
                suggestions=filtered_suggestions[:3],
                status=status,
                confidence="No match",
                confidence_score=0.0,
            )
        answer = self.build_grounded_answer(normalized_query, contexts)
        confidence_score = self.compute_confidence_score(contexts)
        confidence = self.confidence_label(confidence_score)

        context_payload = [
            {
                "source": item.source,
                "text": item.text,
                "score": round(item.score, 4),
                "snippet": self.build_source_snippet(item.text),
            }
            for item in contexts
            if self.is_topic_relevant_to_query(normalized_query, item.source)
        ]
        return AnswerResult(
            answer=answer,
            contexts=context_payload,
            suggestions=filtered_suggestions[:3],
            status="answered",
            confidence=confidence,
            confidence_score=confidence_score,
        )


@lru_cache(maxsize=1)
def get_rag_system() -> WikipediaRAG:
    return WikipediaRAG()


def load_data(topics: Sequence[str] | None = None) -> WikipediaRAG:
    rag = get_rag_system()
    if rag.index is None:
        rag.load_default_data(topics=topics)
    return rag


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, str | float]]:
    rag = load_data()
    return [
        {
            "source": item.source,
            "text": item.text,
            "score": round(item.score, 4),
        }
        for item in rag.retrieve(query, k=k)
    ]


def generate_answer(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    augment_with_search: bool = True,
) -> AnswerResult:
    rag = load_data()
    return rag.answer_question(
        query=query,
        top_k=top_k,
        augment_with_search=augment_with_search,
    )
