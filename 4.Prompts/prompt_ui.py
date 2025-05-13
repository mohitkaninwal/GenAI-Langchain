from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()

model = ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)
template = load_prompt('template.json')

chain = template | model
result = chain.invoke()

print(result)