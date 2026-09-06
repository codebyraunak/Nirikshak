from fastapi import APIRouter
from app.api.audit import router as audit_router

api_router = APIRouter()
api_router.include_router(audit_router, prefix="/audit", tags=["Audit"])
