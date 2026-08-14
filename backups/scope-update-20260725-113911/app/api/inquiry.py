from fastapi import APIRouter, HTTPException, status

from app.models.inquiry import InquiryRequest, InquiryResponse
from app.services.legacy_inquiry_engine import investigate


router = APIRouter(
    prefix="/api",
    tags=["Inquiry Engine"],
)


def run_investigation(payload: InquiryRequest) -> InquiryResponse:
    try:
        result = investigate(
            user_query=payload.query,
            mode=payload.mode,
        )

        return InquiryResponse.model_validate(result)

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Inquiry Engine execution failed.",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        ) from error


@router.post(
    "/cognition/reason",
    response_model=InquiryResponse,
)
def cognition_reason(
    payload: InquiryRequest,
) -> InquiryResponse:
    return run_investigation(payload)


@router.post(
    "/investigate",
    response_model=InquiryResponse,
)
def investigate_alias(
    payload: InquiryRequest,
) -> InquiryResponse:
    return run_investigation(payload)
