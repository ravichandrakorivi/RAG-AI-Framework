import os
import faiss
import pickle
import pdfplumber
import numpy as np
import tiktoken
import re
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
# Hybrid Chunking (SAFE)
# -----------------------------
def chunk_text(text, max_tokens=400, overlap=50):
    enc = tiktoken.encoding_for_model("gpt-4")

    sections = re.split(r'\n(?=[A-Z][A-Za-z ]+:)', text)
    final_chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        tokens = enc.encode(sec)

        if len(tokens) <= max_tokens:
            final_chunks.append(enc.decode(tokens))
        else:
            for i in range(0, len(tokens), max_tokens - overlap):
                chunk = tokens[i:i + max_tokens]
                final_chunks.append(enc.decode(chunk))

    return final_chunks

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
# FINAL Embedding Function (ALL FIXES)
# -----------------------------
def get_embeddings(texts, max_tokens_per_batch=200000, max_items_per_batch=2000):
    enc = tiktoken.encoding_for_model("gpt-4")

    all_embeddings = []
    batch = []
    batch_tokens = 0
    batch_num = 1

    MAX_EMBED_TOKENS = 8000

    i = 0
    while i < len(texts):
        text = texts[i]
        token_list = enc.encode(text)
        tokens = len(token_list)

        # Split oversized chunk
        if tokens > MAX_EMBED_TOKENS:
            print("⚠️ Splitting oversized chunk...")
            sub_chunks = [
                enc.decode(token_list[j:j+400])
                for j in range(0, tokens, 400)
            ]
            texts.extend(sub_chunks)
            i += 1
            continue

        # Check BOTH limits
        if (
            batch_tokens + tokens > max_tokens_per_batch or
            len(batch) >= max_items_per_batch
        ):
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
        i += 1

    # Final batch
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

    faiss.normalize_L2(vectors)

    nlist = 100
    quantizer = faiss.IndexFlatIP(dim)

    index = faiss.IndexIVFFlat(
        quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
    )

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
    print("⚡ Loading index...")
    index = faiss.read_index(INDEX_FILE)

    with open(META_FILE, "rb") as f:
        docs = pickle.load(f)

    return index, docs

# -----------------------------
# Retrieve (HIGH CONTEXT)
# -----------------------------
def retrieve(query, index, docs, k=20):
    q_vec = get_embeddings([query])

    faiss.normalize_L2(q_vec)

    index.nprobe = 20

    D, I = index.search(q_vec, k)

    results = [docs[i] for i in I[0]]

    # Deduplicate
    seen = set()
    unique = []

    for r in results:
        if r["text"] not in seen:
            unique.append(r)
            seen.add(r["text"])

    return unique

# -----------------------------
# 🚆 Troubleshooting Answer
# -----------------------------
def answer_query(query, index, docs):
    results = retrieve(query, index, docs)

    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""
You are an expert locomotive troubleshooting assistant.

Provide a COMPLETE and STRUCTURED answer.

STRICT RULES:
- Do NOT miss any detail
- Include ALL sections:
  Disturbance Text, Protection Activities, Detected Symptom, Causes, Remarks
- List ALL causes clearly
- Preserve formatting
- Be detailed and practical for troubleshooting

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