from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

# load_dotenv()
# DATABASE_URL = os.getenv("DATABASE_URL")  
# print(DATABASE_URL)
engine = create_engine("postgresql://postgres:123@localhost:5432/lyrics_db")
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
