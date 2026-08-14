from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services import memory_service

router = APIRouter(prefix="/api/memory", tags=["Memory"])


@router.get("/notes")
def list_notes():
    return memory_service.get_refined_notes_list()


class SaveNotePayload(BaseModel):
    path: str
    content: str


@router.get("/notes/content")
def get_note_content(path: str):
    try:
        return {"content": memory_service.read_note_content(path)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Note not found")


@router.post("/notes/save")
def save_note(payload: SaveNotePayload):
    memory_service.save_note_content(payload.path, payload.content)
    return {"status": "ok", "path": payload.path}


@router.get("/sessions")
def list_sessions():
    return memory_service.load_session_memory()
