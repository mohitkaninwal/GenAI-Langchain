from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()
model = ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)

schema = [
    ResponseSchema(name='fact_1', description='Fact1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact2 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template  = PromptTemplate(
    template='Give three facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'black hole'})
result = model.invoke(prompt)

final_result= parser.parse(result.content)
print(final_result)