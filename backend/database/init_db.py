from backend.database.connection import engine, Base

# Import models so SQLAlchemy registers them
from backend.database.models import Conversation, Message


def create_tables():

    Base.metadata.create_all(bind=engine)

    print("Database tables created successfully!")


if __name__ == "__main__":
    create_tables()