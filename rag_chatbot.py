import os
from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from dotenv import load_dotenv

# ✅ Load .env variables
load_dotenv()

# ✅ Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ PostgreSQL + pgvector connection
CONNECTION_STRING = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
COLLECTION_NAME = "ai_knowledge_base"  # ✅ Match this with ingest.py

# ✅ Load vectorstore
vectorstore = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embedding_model
)

# ✅ Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ✅ Load FLAN-T5 model
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=300)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# ✅ RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

# ✅ Chat loop
if __name__ == "__main__":
    print("🤖 RAG Chatbot Ready. Type 'exit' to quit.\n")
    while True:
        query = input("👤You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.invoke({"query": query})
        print("🤖Bot:", response["result"])
