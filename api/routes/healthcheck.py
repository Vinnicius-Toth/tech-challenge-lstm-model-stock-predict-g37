from fastapi import APIRouter

router = APIRouter()

@router.get(
    "/",
    summary="Verifica status da API",
    description=(
        "Endpoint de verificação de saúde da aplicação. "
        "Utilizado para monitoramento, testes de disponibilidade "
        "e validação de que a API está ativa."
    ),
    tags=["Healthcheck"]
)
def healthcheck():
    """
    Retorna o status da API.

    - **status**: indica se a API está operando corretamente
    """
    return {"status": "API funcionando corretamente"}