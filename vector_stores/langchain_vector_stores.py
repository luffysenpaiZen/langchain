from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


loader=PyMuPDFLoader(file_path=r'text_splitters\dl_curriculum.pdf')
docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

chunked_docs=splitter.split_documents(docs)

vector_store=Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model='models/embedding-001'),
    persist_directory='chroma_db',
    collection_name='sample'
)

vector_store.add_documents(chunked_docs)

vector_store.get(include=['embeddings','documents', 'metadatas'])



result=vector_store.similarity_search(
    query='what is a deeplearning?',
    k=2
)

print(result)