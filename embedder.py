import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            overlap = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = overlap + " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def build_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embedding_model)
    return db