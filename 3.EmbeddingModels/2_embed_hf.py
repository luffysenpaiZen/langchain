from langchain_huggingface import HuggingFaceEndpointEmbeddings

embedding=HuggingFaceEndpointEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2'
    )

docs=[
    "hi my name is issac",
    "hello your name is gemini"
    "Amaravathi is the capital of Andhra pradesh"
]

result=embedding.embed_documents(docs)

print(str(result))