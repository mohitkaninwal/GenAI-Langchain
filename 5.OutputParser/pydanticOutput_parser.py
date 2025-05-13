from langchain_groq import ChatGroq 
from dotenv import load_dotenv 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model_name='llama-3.1-8b-instant', temperature=0.3)

class Person(BaseModel):
    name:str = Field(description='Name of the person')
    age:str = Field(gt=18,description='Age of the person')
    city:str = Field(description='Name of the city where the person belongs')

parser= PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name , age and city name of a random person living in {place}',
    input_variables=['place'],
    partial_variables={'format instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({'place':'Japenese'})

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | model | parser
final_result = chain.invoke({'place':'India'})
print(final_result)