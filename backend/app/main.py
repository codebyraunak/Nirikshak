from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.audit import router as audit_router
from app.api.semantic import router as semantic_router

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AI-Augmented Vendor-Agnostic Network Security Auditor",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    audit_router,
    prefix=settings.API_V1_STR,
)
app.include_router(
    semantic_router,
    prefix=settings.API_V1_STR,
)

@app.get("/")
def root():
    return {
        "name": "HEXA-FORGE Backend",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }