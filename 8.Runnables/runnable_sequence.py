from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model= ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)

prompt1 = PromptTemplate(
    template='Generate a joke on {topic}',
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template='Explain the logic of this - {text}',
    input_variables=['text']
)

parser =  StrOutputParser()

chain =  RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result =  chain.invoke({'topic':'AI'})

print(result)