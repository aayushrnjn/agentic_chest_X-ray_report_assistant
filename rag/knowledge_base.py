"""Text RAG over the reference notes, built with LangChain.

Build once:   python -m rag.knowledge_base
Then:         HybridRetriever().search("pleural line no lung markings", finding="Pneumothorax")

Pipeline:  markdown -> split by '##' headers (one section per finding, header kept as metadata)
           -> chunks -> embeddings -> FAISS (dense)  +  BM25 (keyword)
           -> Reciprocal Rank Fusion -> cross-encoder rerank -> top k
"""
import pickle
import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import EMBED_MODEL, KB_CHUNKS, KB_INDEX_DIR, KNOWLEDGE_DIR, RERANK_MODEL


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks() -> list[Document]:
    header_split = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "topic")])
    size_split = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = []
    for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
        for section in header_split.split_text(md.read_text()):
            topic = section.metadata.get("topic")
            if not topic:            # skip the file intro before the first '##'
                continue
            for i, piece in enumerate(size_split.split_text(section.page_content)):
                chunk_id = f"KB:{topic}#{i}"
                chunks.append(Document(
                    page_content=f"{topic.replace('_', ' ')}: {piece}",
                    metadata={"id": chunk_id, "topic": topic, "source": md.name},
                ))
    return chunks


def build() -> None:
    chunks = load_chunks()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
    FAISS.from_documents(chunks, embeddings).save_local(str(KB_INDEX_DIR))
    with open(KB_CHUNKS, "wb") as f:
        pickle.dump(chunks, f)
    print(f"indexed {len(chunks)} chunks -> {KB_INDEX_DIR}")


class HybridRetriever:
    """Dense + keyword search, fused and reranked.

    Why hybrid: embeddings catch meaning ("collapsed lung" ~ pneumothorax),
    BM25 catches exact terms (finding names, "Kerley B lines", "CURB-65").
    Why rerank: a cross-encoder reads query and chunk together, which is
    slower but much more accurate than comparing two separate embeddings,
    so we only run it on the ~10 fused candidates.
    """

    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
        self.vs = FAISS.load_local(str(KB_INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        with open(KB_CHUNKS, "rb") as f:
            self.chunks: list[Document] = pickle.load(f)
        self.bm25 = BM25Okapi([_tokenize(d.page_content) for d in self.chunks])
        self.reranker = CrossEncoder(RERANK_MODEL)

    def search(self, query: str, k: int = 3, topic: str | None = None, candidates: int = 10) -> list[dict]:
        # 1) dense retrieval (optionally filtered to one finding's section)
        dense = self.vs.similarity_search(query, k=candidates, filter={"topic": topic} if topic else None)

        # 2) keyword retrieval
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        sparse = [self.chunks[i] for i in order if not topic or self.chunks[i].metadata["topic"] == topic][:candidates]

        # 3) Reciprocal Rank Fusion: score = sum 1/(60 + rank) over both lists
        fused, docs = {}, {}
        for ranked in (dense, sparse):
            for rank, d in enumerate(ranked):
                cid = d.metadata["id"]
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
                docs[cid] = d
        top = sorted(fused, key=fused.get, reverse=True)[:candidates]
        if not top:
            return []

        # 4) cross-encoder rerank
        rr = self.reranker.predict([(query, docs[c].page_content) for c in top])
        ranked = sorted(zip(top, rr), key=lambda t: -t[1])[:k]
        return [{"id": c, "topic": docs[c].metadata["topic"], "score": round(float(s), 3),
                 "text": docs[c].page_content} for c, s in ranked]


if __name__ == "__main__":
    build()
