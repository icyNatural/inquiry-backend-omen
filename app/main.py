from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.investigations import router as investigations_router
from app.api.inquiry import router as inquiry_router
from app.api import memory as memory_router
from app.api import scopes as scopes_router


app = FastAPI(
    title="Inquiry Backend",
    description="Backend service for Inquiry Observatory.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(investigations_router)
app.include_router(inquiry_router)
app.include_router(memory_router.router)
app.include_router(scopes_router.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "inquiry-backend",
    }