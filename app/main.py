from fastapi import FastAPI

app = FastAPI(
    title="Inquiry Backend",
    description="Backend service for Inquiry Observatory.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "inquiry-backend",
    }
