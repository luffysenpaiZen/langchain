from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

loader=PyMuPDFLoader(file_path=r'text_splitters\dl_curriculum.pdf')

docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20
)

result=splitter.split_documents(docs)

print(result)
print(len(result))