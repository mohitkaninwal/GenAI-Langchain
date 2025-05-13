from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)

passthrough = RunnablePassthrough()
parser =  StrOutputParser()

prompt = PromptTemplate(
    template='Generate a joke on {topic}',
    input_variables=['topic']
)

def word_counter(text):
    return len(text.split())

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_counter)
})

final_chain =  RunnableSequence(joke_gen_chain,parallel_chain)

response  = final_chain.invoke({'topic':'Unemployment in India'})
print(response['word_count'])