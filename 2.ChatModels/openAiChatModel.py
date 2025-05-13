from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
result = model.invoke('How to identify if the HDMI cable is broken while projecting to the smart tv')
print(result.content)