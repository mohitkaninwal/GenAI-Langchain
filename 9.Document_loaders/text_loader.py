from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model= ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)

prompt = PromptTemplate(
    template='Write a summary of the {poem}',
    input_variables=['poem']
)

loader = TextLoader('example.txt',encoding='utf-8')
docs = loader.load()

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({'poem':docs[0].page_content})

print(response)