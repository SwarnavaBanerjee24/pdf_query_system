import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def init_astra():
    cassio.init(
        token=Config.ASTRA_DB_APPLICATION_TOKEN, 
        database_id=Config.ASTRA_DB_ID
    )
    embedding = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    vector_store = Cassandra(
        embedding=embedding,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None
    )
    return vector_store

def process_and_store_text(text: str, vector_store):
    """Split text into chunks and store in vector database"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(text)
    chunks_to_add = min(50, len(texts))
    vector_store.add_texts(texts[:chunks_to_add])
    return chunks_to_add