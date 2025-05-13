from langchain_groq import ChatGroq
from typing import TypedDict, Annotated,Optional
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)

#schema
class Review(TypedDict):
    key_themes:Annotated[list[str],'Write down all the key themes discussed in the review in a list']
    summary: Annotated[str, 'A brief summary of the review']
    sentiment: Annotated[str, 'Return sentiment of the review either positive, negative or neutral']
    pros: Annotated[Optional[list[str]], 'write down all the pros inside a list']
    cons: Annotated[Optional[list[str]], 'write down all the cons inside a list']
    name: Annotated[str,'Write the name of the reviewer']

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware feels amazing but the software feels bloated. The new bugs have been introduced in the latest software update. I feel frustated due to the latest features which are not user friendly.
                                 
pros:Good hardware
cons: Bloated software

Review by Mohit
                        """)

print(result['pros'])       