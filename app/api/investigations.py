from fastapi import APIRouter, HTTPException, status

from app.models.investigation import (
    Investigation,
    InvestigationCreate,
    InvestigationUpdate,
)
from app.repositories.investigation_repository import InvestigationRepository


router = APIRouter(prefix="/api/investigations", tags=["Investigations"])
repository = InvestigationRepository()


@router.post(
    "",
    response_model=Investigation,
    status_code=status.HTTP_201_CREATED,
)
def create_investigation(payload: InvestigationCreate) -> Investigation:
    return repository.create(payload)


@router.get("", response_model=list[Investigation])
def list_investigations() -> list[Investigation]:
    return repository.list_all()


@router.get("/{investigation_id}", response_model=Investigation)
def get_investigation(investigation_id: str) -> Investigation:
    investigation = repository.get(investigation_id)

    if investigation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation not found",
        )

    return investigation


@router.patch("/{investigation_id}", response_model=Investigation)
def update_investigation(
    investigation_id: str,
    payload: InvestigationUpdate,
) -> Investigation:
    try:
        investigation = repository.update(investigation_id, payload)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error

    if investigation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation not found",
        )

    return investigation


@router.delete(
    "/{investigation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_investigation(investigation_id: str) -> None:
    deleted = repository.delete(investigation_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation not found",
        )


@router.get("/{investigation_id}/children", response_model=list[Investigation])
def list_children(investigation_id: str) -> list[Investigation]:
    children = repository.list_children(investigation_id)
    return children