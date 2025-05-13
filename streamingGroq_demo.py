from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

chat = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
prompt = ChatPromptTemplate([('human','write briefly about {topic} in 1000 words')])

chain = prompt | chat

for chunk in chain.stream({"topic": "Fidel castro"}):
    print(chunk.content, end="", flush=True)