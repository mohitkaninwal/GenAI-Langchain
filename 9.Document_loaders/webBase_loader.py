from langchain_community.document_loaders import WebBaseLoader

url=''

loader = WebBaseLoader()
docs = loader.load()