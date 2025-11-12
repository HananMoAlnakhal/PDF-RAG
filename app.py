# app.py
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import re
from typing import List, Dict

st.set_page_config(layout="wide")
st.title("PDF QA with Highlighting — Prototype")

# ---------------------------
# Utilities
# ---------------------------
def extract_pdf_texts(file) -> List[Dict]:
    """
    Extracts text per page and returns list of dicts:
    [{ 'doc_id': id, 'filename': name, 'page': pageno, 'text': text_on_page }]
    """
    out = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            out.append({"doc_id": id(file), "filename": file.name, "page": i+1, "text": text})
    return out

def chunk_text(text: str, chunk_size=800, overlap=200):
    """
    Simple char-based chunking that returns list of (chunk_text, start_char, end_char)
    """
    chunks = []
    start = 0
    L = len(text)
    if L == 0:
        return []
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

def simple_highlight(chunk_text: str, query: str) -> str:
    """
    Highlights occurrences of query words in chunk_text by wrapping them in <mark>.
    Very simple: splits query into words and highlights occurrences (case-insensitive).
    """
    if not query.strip():
        return chunk_text.replace("\n", "<br>")
    qwords = [re.escape(w) for w in re.split(r"\s+", query.strip()) if w]
    if not qwords:
        return chunk_text.replace("\n", "<br>")
    pattern = re.compile(r"(" + r"|".join(qwords) + r")", flags=re.IGNORECASE)
    highlighted = pattern.sub(r"<mark>\1</mark>", chunk_text)
    return highlighted.replace("\n", "<br>")

# ---------------------------
# Sidebar: settings & model load
# ---------------------------
st.sidebar.header("Settings / Models")
embed_model_name = st.sidebar.text_input("Embedding model", value="all-MiniLM-L6-v2")
chunk_size = st.sidebar.number_input("Chunk size (chars)", min_value=200, max_value=5000, value=800, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=50)
top_k = st.sidebar.number_input("Top-K retrieved chunks", min_value=1, max_value=20, value=5)
use_generator = st.sidebar.checkbox("Synthesize concise answer (Flan-T5 small)", value=True)

@st.cache_resource(show_spinner=False)
def load_embedding_model(name):
    return SentenceTransformer(name)

embed_model = load_embedding_model(embed_model_name)

generator_pipeline = None
if use_generator:
    with st.spinner("Loading generator model (this may take a bit)..."):
        # Flan-T5-small or you can change to other text2text model
        generator_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

# ---------------------------
# Upload PDFs
# ---------------------------
uploaded = st.file_uploader("Upload PDF files (multiple allowed)", accept_multiple_files=True, type=["pdf"])

if uploaded:
    st.info(f"{len(uploaded)} files uploaded. Extracting text...")
    # Extract text per page for all PDFs
    pages = []  # list of dicts with doc/page/text
    for f in uploaded:
        pages.extend(extract_pdf_texts(f))
    st.success(f"Extracted text from {len(pages)} pages across {len(uploaded)} documents.")

    # Create chunks with metadata
    chunk_texts = []
    metadatas = []  # parallel list for metadata
    for p in pages:
        text = p["text"]
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        for chunk, start, end in chunks:
            # save mapping to original doc/page and character offsets
            metadata = {
                "doc_id": p["doc_id"],
                "filename": p["filename"],
                "page": p["page"],
                "char_start": start,
                "char_end": end,
            }
            chunk_texts.append(chunk)
            metadatas.append(metadata)

    if len(chunk_texts) == 0:
        st.error("No text found in uploaded PDFs (maybe they are scanned images). Use OCR or upload selectable-text PDFs.")
    else:
        st.write(f"Total chunks: {len(chunk_texts)}")

        # Compute embeddings
        with st.spinner("Computing embeddings..."):
            embeddings = embed_model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # use inner product on normalized vectors
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        # Save normalized embeddings (for retrieval score computation)
        norm_emb = embeddings  # already normalized

        st.success("Index built. Ask a question below!")

        # ---------------------------
        # Query interface
        # ---------------------------
        st.markdown("---")
        st.header("Ask a question")
        query = st.text_area("Question", height=80)
        btn = st.button("Search")

        if btn and query.strip():
            with st.spinner("Retrieving relevant passages..."):
                q_emb = embed_model.encode([query], convert_to_numpy=True)
                faiss.normalize_L2(q_emb)
                D, I = index.search(q_emb, top_k)
                scores = D[0]
                idxs = I[0]

            retrieved = []
            for score, idx in zip(scores, idxs):
                if idx < 0:
                    continue
                meta = metadatas[idx]
                chunk = chunk_texts[idx]
                # prepare highlighted HTML
                highlighted = simple_highlight(chunk, query)
                retrieved.append({
                    "score": float(score),
                    "filename": meta["filename"],
                    "page": meta["page"],
                    "chunk": chunk,
                    "highlighted_html": highlighted
                })

            # Display retrieved results and which document
            st.subheader("Retrieved passages (source and page shown)")
            for i, r in enumerate(retrieved):
                st.markdown(f"**Rank {i+1}** — score: `{r['score']:.3f}` — **{r['filename']}** — page **{r['page']}**")
                st.markdown(r["highlighted_html"], unsafe_allow_html=True)
                st.markdown("---")

            # Optionally synthesize final answer from the top chunks
            if use_generator and generator_pipeline is not None:
                with st.spinner("Synthesizing concise answer from retrieved passages..."):
                    # Build a prompt: include top passages as context
                    context = "\n\n".join([f"Document: {r['filename']} (page {r['page']})\n{r['chunk']}" for r in retrieved])
                    # Keep prompt small — flan-t5-small has token limits; shorten if too long
                    prompt = f"Answer the question using the context below. If the answer is not in the context, say 'Not found in the provided documents'.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer concisely:"
                    out = generator_pipeline(prompt, max_length=256, do_sample=False)
                    answer = out[0]["generated_text"]
                st.subheader("Generated answer")
                st.write(answer)

            # Helpful: show list of documents that contained the top results
            doc_set = []
            for r in retrieved:
                if r["filename"] not in doc_set:
                    doc_set.append(r["filename"])
            st.subheader("Documents that contained top answers")
            for d in doc_set:
                st.write(f"- {d}")

            # Done
            st.success("Done - passages highlighted above. Click any passage to copy if you need.")