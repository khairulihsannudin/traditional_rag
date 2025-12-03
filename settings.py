# settings.py - Traditional RAG Configuration
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Environment Variables ---
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")

# Langchain Tracing (Optional)
os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "")
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get("LANGCHAIN_ENDPOINT", "")

# --- LLM init ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.environ["GOOGLE_API_KEY"])
# llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
# llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")

print(f"Using LLM: {llm.model_dump()}")
