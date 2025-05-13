from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)
prompt1 =  PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 =  PromptTemplate(
    template='Generate a five point summary on the \n {text}',
    input_variables=['text']
)

parser= StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

final_result = chain.invoke({'topic':'sun'})
print(final_result)

