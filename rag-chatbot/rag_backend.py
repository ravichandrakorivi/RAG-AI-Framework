import os
import faiss
import pickle
import pdfplumber
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_FOLDER = "pdfs"
INDEX_FILE = "faiss_ivf.index"
META_FILE = "metadata.pkl"

EMBED_MODEL = "text-embedding-3-large"

# -----------------------------
# Chunking (smaller = faster + cheaper)
# -----------------------------
def chunk_text(text, max_tokens=300, overlap=50):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunks.append(enc.decode(tokens[i:i + max_tokens]))

    return chunks

# -----------------------------
# Load PDFs
# -----------------------------
def load_pdfs():
    docs = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            print(f"📄 Reading: {file}")

            with pdfplumber.open(os.path.join(PDF_FOLDER, file)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""

            chunks = chunk_text(text)
            print(f"   → {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                docs.append({
                    "text": chunk,
                    "source": file,
                    "chunk_id": i
                })

    print(f"\n✅ Total chunks: {len(docs)}")
    return docs

# -----------------------------
# Token-based batching (FIX 🚀)
# -----------------------------
def get_embeddings(texts, max_tokens_per_batch=200000):
    enc = tiktoken.encoding_for_model("gpt-4")

    all_embeddings = []
    batch = []
    batch_tokens = 0
    batch_num = 1

    for text in texts:
        tokens = len(enc.encode(text))

        if batch_tokens + tokens > max_tokens_per_batch:
            print(f"🔄 Embedding batch {batch_num} ({len(batch)} chunks)")

            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch
            )

            all_embeddings.extend([e.embedding for e in response.data])

            batch = []
            batch_tokens = 0
            batch_num += 1

        batch.append(text)
        batch_tokens += tokens

    # final batch
    if batch:
        print(f"🔄 Embedding batch {batch_num} ({len(batch)} chunks)")

        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )

        all_embeddings.extend([e.embedding for e in response.data])

    return np.array(all_embeddings).astype("float32")

# -----------------------------
# Build FAISS IVF (Cosine)
# -----------------------------
def build_index(docs):
    texts = [d["text"] for d in docs]

    print("\n🚀 Generating embeddings...")
    vectors = get_embeddings(texts)

    dim = vectors.shape[1]

    # Normalize → cosine similarity
    faiss.normalize_L2(vectors)

    # IVF index
    nlist = 100
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    print("🔄 Training FAISS index...")
    index.train(vectors)

    print("📦 Adding vectors...")
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(docs, f)

    print("✅ Index saved!")

# -----------------------------
# Load index
# -----------------------------
def load_index():
    print("⚡ Loading FAISS index...")
    index = faiss.read_index(INDEX_FILE)

    with open(META_FILE, "rb") as f:
        docs = pickle.load(f)

    return index, docs

# -----------------------------
# Retrieve
# -----------------------------
def retrieve(query, index, docs, k=5):
    q_vec = get_embeddings([query])

    faiss.normalize_L2(q_vec)

    index.nprobe = 10

    D, I = index.search(q_vec, k)

    return [docs[i] for i in I[0]]

# -----------------------------
# Answer
# -----------------------------
def answer_query(query, index, docs):
    results = retrieve(query, index, docs)

    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, results