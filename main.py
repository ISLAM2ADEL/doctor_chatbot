from config import NGROK_AUTH_TOKEN
from model_loader import load_model
from pdf_loader import load_pdf_text
from embedder import chunk_text, build_vector_store
from llm_wrapper import SimpleHuggingFaceLLM
from api import setup_api
from pyngrok import ngrok
import uvicorn

# تحميل الموديل والتوكن
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model, tokenizer = load_model(model_name)

llm = SimpleHuggingFaceLLM(model=model, tokenizer=tokenizer)

# تحميل النصوص من PDF
pdf_text = load_pdf_text("Oxford-Handbook-of-Medical-Dermatology.pdf")
chunks = chunk_text(pdf_text)
db = build_vector_store(chunks)
retriever = db.as_retriever()

# إعداد FastAPI
app = setup_api(llm, retriever)

# Ngrok
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(8000)
print("✅ Your API is available at:", public_url)

# شغل السيرفر
if __name__ == "__main__":
    uvicorn.run(app, port=8000)