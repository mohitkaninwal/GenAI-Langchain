from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch,RunnableLambda

load_dotenv()

model = ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)

class Feedback(BaseModel):
    sentiment: Literal['positive','negative']= Field(description='Give sentiment of the feedback')

parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the {feedback} into positive or negative.\n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

classifier_chain= prompt1 | model | parser2
chain1 = prompt2 | model | parser 
chain2= prompt3 | model | parser

branch_chain= RunnableBranch(
    (lambda x:x.sentiment=='positive', chain1),       # condition, chain
    (lambda x:x.sentiment=='negative', chain2),       # condition, chain
    RunnableLambda(lambda x: 'Could not find sentiment') # default chain using runnable lambda
)

chain = classifier_chain | branch_chain

response  = chain.invoke({'feedback':'this is amazing laptop'})
print(response)