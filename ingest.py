from langchain_community.document_loaders import (
    TextLoader, PDFPlumberLoader, UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from dotenv import load_dotenv
import os

# ✅ Load environment variables from .env file
load_dotenv()

# 1. Folder containing your documents
docs_folder = "./docs"

# 2. Load all documents from the folder
all_docs = []
for file in os.listdir(docs_folder):
    path = os.path.join(docs_folder, file)
    if file.endswith(".txt"):
        loader = TextLoader(path)
    elif file.endswith(".pdf"):
        loader = PDFPlumberLoader(path)
    elif file.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(path)
    else:
        print(f"⛔ Skipped unsupported file: {file}")
        continue
    documents = loader.load()
    all_docs.extend(documents)

print(f"✅ Loaded {len(all_docs)} documents")

# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_chunks = splitter.split_documents(all_docs)
print(f"✅ Split into {len(docs_chunks)} chunks")

# 4. Initialize HuggingFace embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. PGVector DB connection details
CONNECTION_STRING = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"


# ✅ 6. Clean null characters
cleaned_chunks = []
for doc in docs_chunks:
    doc.page_content = doc.page_content.replace('\x00', '')
    cleaned_chunks.append(doc)

# 7. Store into PGVector
PGVector.from_documents(
    documents=cleaned_chunks,
    embedding=embedding,
    collection_name="ai_knowledge_base",
    connection_string=CONNECTION_STRING,
)

print("✅✅ All documents embedded and stored in PGVector successfully!")

