from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
 Answer this query
"""
)

template.save('template.json')