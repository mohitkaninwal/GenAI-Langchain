from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv 

load_dotenv()

model = ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)

prompt = PromptTemplate(
    template='Generate five facts about {topic}',
    input_variables=['topic']
)

parser= StrOutputParser()
chain = prompt | model | parser
result =  chain.invoke({'topic':'India'})

print(result)

