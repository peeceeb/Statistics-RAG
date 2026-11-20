
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.config.settings import DATABASE_URL


# --------------------------------------------------
# Database Configuration
# --------------------------------------------------

INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME", "Enterprise_RAG")

DB_IAM_AUTH = (
    os.getenv("DB_IAM_AUTH", "false").lower() == "true"
)

IS_CLOUD_SQL = bool(INSTANCE_CONNECTION_NAME)


# --------------------------------------------------
# SQLAlchemy Engine
# --------------------------------------------------

if IS_CLOUD_SQL:

    from google.cloud.sql.connector import Connector, IPTypes

    connector = Connector()

    def getconn():
        return connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
            enable_iam_auth=DB_IAM_AUTH,
            ip_type=IPTypes.PUBLIC
        )

    engine = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False
    )

else:

    if not DATABASE_URL:
        raise ValueError(
            "DATABASE_URL is required for local database connections."
        )

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


# --------------------------------------------------
# Connector Lifecycle
# --------------------------------------------------

def close_database_connections():

    engine.dispose()

    if IS_CLOUD_SQL:
        connector.close()
