from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace
import os
from langchain.schema.runnable import RunnableParallel

load_dotenv()


model1 = ChatHuggingFace(
    model_name="deepseek-ai/DeepSeek-Prover-V2-671B",
    huggingfacehub_api_token=os.getenv("ACCESS_TOKEN")
)
model2 = (
    
)

prompt1= PromptTemplate(
    template=''
)
prompt2= PromptTemplate(
    template=''
)
prompt3= PromptTemplate(
    template=''
)
parser = StrOutputParser()

parallel_chain= RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

text = """Some random text
"""

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'text':text})
print(result)