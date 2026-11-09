
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase


# Load environment variables
load_dotenv()

# Read PostgreSQL connection URL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from .env")


# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False
)


# Create database session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


# Base class for database models
class Base(DeclarativeBase):
    pass


# FastAPI database dependency
def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# Test database connection
def test_db_connection():

    with engine.connect() as connection:

        result = connection.execute(text("SELECT 1"))

        print("PostgreSQL connection successful!")
        print("Test result:", result.scalar())
