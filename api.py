from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
import re

app = FastAPI()

class DoctorQuestion(BaseModel):
    message: str
    translated_conversation: str

few_shot_prompt = """
You are a highly knowledgeable and concise dermatology assistant, working alongside a dermatologist.
Base your answers strictly on the medical context provided either in the patient conversation or the RAAG medical reference.
Never guess or hallucinate. Your answers must be short, direct, and medically accurate.

If a question is unrelated to dermatology or medicine, reply with:
**"I'm specialized in dermatology and cannot assist with non-medical topics."**

Here is the previous conversation between a patient and the assistant:
---
{translated_conversation}
---

Here is a relevant medical reference (RAAG):
---
{raag_reference}
---

Doctor's Question: {question}
Answer:
"""

template = PromptTemplate(
    template=few_shot_prompt,
    input_variables=["translated_conversation", "raag_reference", "question"]
)

def clean_text(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text).strip()
    text = re.sub(r'Prompt after formatting:.*?\n', '', text, flags=re.DOTALL)
    match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)
    answer = match.group(1).strip() if match else text
    return "\n".join(dict.fromkeys(answer.split("\n")))

@app.get("/")
def root():
    return {"message": "Dermatology Assistant API with RAAG is running."}

def setup_api(llm, retriever):
    @app.post("/ask")
    def ask(msg: DoctorQuestion):
        user_input = msg.message.strip()
        translated_conversation = msg.translated_conversation.strip()

        try:
            retrieved_docs = retriever.get_relevant_documents(user_input)
            raag_context = "\n".join([doc.page_content for doc in retrieved_docs])

            prompt = template.format(
                translated_conversation=translated_conversation,
                raag_reference=raag_context,
                question=user_input
            )

            result = llm(prompt)
            cleaned = clean_text(result)
            return {"response": cleaned}
        except Exception as e:
            return {"error": str(e)}

    return app