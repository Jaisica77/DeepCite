import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routes import upload, rag, evaluate
from routes.agent import router as agent_router

os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/embeddings", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting DeepCite...")
    print("Data directories ready")
    yield
    print("Shutting down...")


app = FastAPI(
    title="DeepCite API",
    description="Agentic RAG platform with document and web citations",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        "https://deepcite.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(rag.router, prefix="/api", tags=["RAG"])
app.include_router(evaluate.router, prefix="/api", tags=["Evaluate"])
app.include_router(agent_router, prefix="/api", tags=["Agent"])


@app.get("/")
def root():
    return {
        "message": "DeepCite API",
        "version": "2.0.0",
        "endpoints": {
            "upload": "POST /api/upload-docs",
            "list_docs": "GET /api/docs",
            "run_rag": "POST /api/run-rag",
            "evaluate": "POST /api/evaluate",
            "agent_chat": "POST /api/agent-chat",
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
