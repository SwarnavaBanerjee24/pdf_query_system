import os
import cassio
import openai
from datasets import load_dataset
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
import textwrap

# Load credentials from environment variables
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not ASTRA_DB_APPLICATION_TOKEN or not ASTRA_DB_ID or not OPENAI_API_KEY:
    raise ValueError("Missing credentials! Set ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_ID, and OPENAI_API_KEY as environment variables.")

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Initialize Astra DB connection
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize LangChain components
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,  # Explicit session setup can be added if needed
    keyspace=None,
)

# Extract text from multiple PDFs
pdf_directory = os.path.join(ROOT_DIRECTORY, "SOURCE_DOCUMENTS")
raw_text = ""

for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        raw_text += extract_text(pdf_path) + "\n\n"  # Extract and concatenate text

if not raw_text.strip():
    raise ValueError("No valid text extracted from PDFs.")

# Split text into chunks using a better splitting strategy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    length_function=len
)
texts = text_splitter.split_text(raw_text)

# Add text chunks to vector store
astra_vector_store.add_texts(texts[:50])  # Limiting insertion for now
print(f"Inserted {len(texts[:50])} text chunks.")

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Function to format answers neatly
def format_answer(answer):
    return textwrap.fill(answer, width=80)

# Interactive Q&A loop
first_question = True

while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit, 'reset' to start over): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit, 'reset' to start over): ").strip()

    if query_text.lower() == "quit":
        break
    elif query_text.lower() == "reset":
        first_question = True
        print("\nConversation reset.")
        continue

    if not query_text:
        continue

    first_question = False
    print("\nQUESTION:", query_text)

    # Retrieve documents before calling OpenAI to reduce API costs
    retrieved_docs = astra_vector_store.similarity_search(query_text, k=4)

    if retrieved_docs:
        context = " ".join([doc.page_content for doc in retrieved_docs])
        prompt = [
            {"role": "system", "content": "Use the following context to answer the question:"},
            {"role": "user", "content": context + "\n" + query_text}
        ]
        answer = llm.generate(prompt).strip()
    else:
        answer = "No relevant documents found."

    print("\nANSWER:\n", format_answer(answer))

    # Show relevant documents
    print("\nTOP RELEVANT DOCUMENTS:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print(f"    [{score:.4f}] \"{doc.page_content[:84]} ...\"")
