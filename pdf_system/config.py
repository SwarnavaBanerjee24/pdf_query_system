from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
    
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        if not cls.ASTRA_DB_APPLICATION_TOKEN or not cls.ASTRA_DB_ID:
            raise ValueError("Astra DB credentials not found in environment variables")