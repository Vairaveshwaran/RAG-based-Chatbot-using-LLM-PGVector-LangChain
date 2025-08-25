# RAG-based-Chatbot-using-LLM-PGVector-LangChain
This project implements a Retrieval-Augmented Generation (RAG) Chatbot powered by:

LLMs (Large Language Models, e.g., FLAN-T5)

LangChain for orchestration

PGVector (PostgreSQL extension) for vector storage

It allows you to:

Upload local documents (.txt, .pdf, .docx).

Automatically embed them into a vector database.

Ask natural language questions, and get answers based on your document content.

✨ Features

📂 Multi-format ingestion → Supports .txt, .pdf, and .docx files.

🧠 Context-aware Q&A → Answers based only on your uploaded content.

🗄️ Vector DB with PGVector → Stores embeddings for fast retrieval.

🔗 LangChain-powered pipeline → Orchestrates document loading, embedding, and retrieval.

🐳 Docker support → PostgreSQL + PGVector run easily in a container.

⚙️ Project Flow

Ingest documents

ingest.py loads .txt, .pdf, .docx.

Splits into chunks and generates embeddings.

Stores embeddings in PostgreSQL with PGVector.

Ask questions

rag_chatbot.py retrieves the most relevant chunks.

Passes them to the LLM (FLAN-T5).

Generates an accurate, context-based answer.

🛠️ Setup
1️⃣ Clone the repo
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Start PostgreSQL with PGVector (via Docker)
docker-compose up -d

5️⃣ Configure environment variables

Copy .env.example → .env and set your values:

DB_NAME=rag_chat
DB_USER=postgres
DB_PASS=your_password_here
DB_HOST=localhost
DB_PORT=5432

6️⃣ Ingest documents

Put your files in the data/ folder, then run:

python ingest.py

7️⃣ Ask questions
python rag_chatbot.py

📂 Project Structure
rag-chatbot/
│-- ingest.py          # Script to load docs + create embeddings
│-- rag_chatbot.py     # Chatbot for Q&A using LLM + vector DB
│-- docker-compose.yml # Runs PostgreSQL + PGVector
│-- .env.example       # Example environment variables
│-- requirements.txt   # Dependencies
│-- README.md          # Documentation
│-- Notess.txt         # Setup notes (optional)

🔮 Future Improvements

Add support for more document types (CSV, PPTX).

Integrate with OpenAI GPT models for more powerful answers.

Build a web UI (Streamlit/Flask) for interactive usage.

Add email/PDF export of chat responses.
