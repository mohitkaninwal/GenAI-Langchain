from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Annotated,TypedDict,Optional,Literal
from pydantic import BaseModel, Field

load_dotenv()

model= ChatGroq(model_name='llama-3.1-8b-instant',temperature=0.3)

#schema
class Review(BaseModel):
    key_themes:list[str]=Field(description='Write down all the key themes discussed in the review in a list')
    summary: str = Field(description='A brief summary of the review')
    sentiment: Literal['pos','neg'] = Field(description='Return sentiment of the review either positive, negative or neutral')
    pros: Optional[list[str]]=Field(default=None,description='write down all the pros inside a list')
    cons: Optional[list[str]]=Field(default=None,description='write down all the cons inside a list')
    name: Optional[str]=Field(default=None,description='Write the name of the reviewer')

structured_model=model.with_structured_output(Review)

result = structured_model.invoke("""The hardware feels amazing but the software feels bloated. The new bugs have been introduced in the latest software update. I feel frustated due to the latest features which are not user friendly.
                                 
pros:Good hardware
cons: Bloated software

Review by Mohit
                        """)

print(result.name)