from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
import uvicorn
import os
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

## Env Variable
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

app = FastAPI(
    title="LangChain Server",
    version= "1.0",
    description="A Simple API Server"
    )



llm1 = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm2 = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} with rhyming words")

add_routes(
    app,
    prompt1 | llm1,
    path = "/essay"
)

add_routes(
    app,
    prompt2 | llm2,
    path = "/poem"
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)