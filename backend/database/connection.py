
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.config.settings import DATABASE_URL


# --------------------------------------------------
# Database Configuration
# --------------------------------------------------

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from .env")


# --------------------------------------------------
# SQLAlchemy Engine
# --------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False
)


# --------------------------------------------------
# Database Session Factory
# --------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


# --------------------------------------------------
# Base Class for Database Models
# --------------------------------------------------

class Base(DeclarativeBase):
    pass


# --------------------------------------------------
# FastAPI Database Dependency
# --------------------------------------------------

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# --------------------------------------------------
# Test Database Connection
# --------------------------------------------------

def test_db_connection():

    with engine.connect() as connection:

        result = connection.execute(text("SELECT 1"))

        print("PostgreSQL connection successful!")
        print("Test result:", result.scalar())
