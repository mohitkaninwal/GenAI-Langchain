from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda,RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)

passthrough = RunnablePassthrough()
parser =  StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='summarize the following text {topic}',
    input_variables=['topic']
)

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)
final_chain= RunnableSequence(report_gen_chain,branch_chain)
response = final_chain.invoke({'topic':'India vs Pakistan'})
print(response)