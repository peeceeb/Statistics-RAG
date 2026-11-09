from fastapi import FastAPI

from backend.routes.chat import router as chat_router


app = FastAPI(
    title="RAG API",
    description="Backend API for RAG AI Assistant",
    version="1.0.0"
)


# Register chat routes
app.include_router(chat_router)


@app.get("/")
def root():

    return {
        "message": "RAG API is running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }