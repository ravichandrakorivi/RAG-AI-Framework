import os
import pdfplumber
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key = os.getenv("OPEN_API_KEY"))

PDF_FOLDER = "pdfs"
OUTPUT_FOLDER = "outputs"

# ------------------------------
# Chunk long text safely
# ------------------------------
def chunk_text(text, max_tokens=12000):
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = enc.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = enc.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)

    return chunks

# ------------------------------
# Extract text from PDF
# ------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text

# ------------------------------
# Generate MCQs for a chunk
# ------------------------------
def generate_mcqs_from_chunk(chunk_text):
    prompt = f"""
You are an expert textbook instructor.

Based on the following textbook content, generate MCQs in **both English and Hindi**, in the EXACT FORMAT below:

Q<n> – <English Question>
Q<n> – <Hindi Question>
a) <OptionA_English> / <OptionA_Hindi>
b) <OptionB_English> / <OptionB_Hindi>
c) <OptionC_English> / <OptionC_Hindi>
d) <OptionD_English> / <OptionD_Hindi>
Correct Answer: <a/b/c/d>

Requirements:
- Generate
    - A list of topics covered
    - For each topic:
        - **5 MCQs (4 options each)**.
- Hindi must be natural and grammatically correct.
- Options must be meaningful.
- "Correct Answer:" MUST appear exactly like this.
- No explanations.
- Output ONLY the MCQs. Nothing else.

TEXT CHUNK:
{chunk_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content

# ------------------------------
# Main processing
# ------------------------------
def process_pdfs():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        print("❌ No PDFs found in 'pdfs/' folder.")
        return

    for pdf in pdf_files:
        print(f"\n📘 Processing: {pdf}")
        pdf_path = os.path.join(PDF_FOLDER, pdf)

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        print(f"➡ Extracted text length: {len(text)} characters")
        print(f"➡ Number of chunks: {len(chunks)}")

        output_file = os.path.join(OUTPUT_FOLDER, f"{pdf}_MCQs.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# MCQs for {pdf}\n\n")

            for i, chunk in enumerate(chunks):
                print(f"   🧩 Processing chunk {i+1}/{len(chunks)}...")
                mcq_output = generate_mcqs_from_chunk(chunk)

                f.write(f"## Chunk {i+1}\n")
                f.write(mcq_output + "\n\n")

        print(f"✅ Completed → Saved to: {output_file}")

process_pdfs()
