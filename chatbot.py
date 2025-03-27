from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
import os

load_dotenv()
groq_api_key = os.environ.get('GROQ_API_KEY')

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
chat_history=[
    SystemMessage(content='You are a helpful AI Assistant')
]

while True:
    user_input= input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI: ',result.content)

print(chat_history)