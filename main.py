from fastapi import FastAPI

app = FastAPI(title="Wild AI Cloud")


@app.get("/")
def home():
    return {
        "project": "Wild AI",
        "status": "online",
        "message": "Cloud server is working"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
