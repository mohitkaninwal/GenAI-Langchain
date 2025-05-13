from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)

passthrough = RunnablePassthrough()
parser =  StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a joke on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a explanation on {topic}',
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2, model , parser)
})

final_chain= RunnableSequence(joke_gen_chain,parallel_chain)
response  = final_chain.invoke({'topic':'cricket'})
print(response)