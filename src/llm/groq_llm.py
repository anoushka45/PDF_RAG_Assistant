import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class GroqLLM:
    def __init__(self, model_name):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0.1
        )

    def generate(self, prompt):
        return self.llm.invoke(prompt).content
