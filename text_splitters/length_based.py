from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader(file_path=r'text_splitters\dl_curriculum.pdf')

docs=loader.load()

splitter=CharacterTextSplitter(chunk_size=200,chunk_overlap=22,separator='')

result=splitter.split_documents(docs)

print(result)