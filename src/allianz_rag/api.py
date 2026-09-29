"""FastAPI transport for the claims assistant."""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

from allianz_rag import __version__
from allianz_rag.models import AnalyzeRequest, AnalyzeResponse
from allianz_rag.service import ClaimsService, get_service

app = FastAPI(
    title="Allianz Claims Copilot API",
    description="Evidence-backed CIDE/ASCIDE/CICOS claim analysis",
    version=__version__,
)

ServiceDependency = Annotated[ClaimsService, Depends(get_service)]


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight liveness response without loading the ML stack."""

    return {"status": "ok", "version": __version__}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    service: ServiceDependency,
) -> AnalyzeResponse:
    """Analyse a question or accident description using grounded evidence."""

    try:
        return service.analyze(request.query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The analysis service is not ready. Check the index and model configuration.",
        ) from exc
