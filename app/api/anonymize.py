"""POST /anonymize: borra cabeças, rostos e placas de uma foto e devolve a imagem borrada.

Quem chama: a API .NET, quando o cidadão manda a foto de uma ocorrência resolvida. A .NET só
guarda no Blob o que volta daqui; se esta rota falhar, a .NET responde 503 e não guarda nada.

Usa o mesmo anonimizador do /analisar (rostos, placas e pessoas). Sem ele, responde 503:
por privacidade, foto sem borrão nunca sai daqui.
"""
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from app.api.analyze import read_upload
from app.api.dependencies import anonymizer_dependency
from app.security import require_api_key
from app.vision.anonymizer import AnonymizationError, Anonymizer
from app.vision.frames import InvalidImageError, decode_image, encode_jpeg

router = APIRouter(tags=["Anonimização"])


@router.post(
    "/anonymize",
    summary="Borra cabeças, rostos e placas de uma foto e devolve a imagem (JPEG)",
    response_class=Response,
    dependencies=[Depends(require_api_key)],
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Foto borrada. Os headers X-Faces-Blurred, X-Plates-Blurred e "
                           "X-Persons-Blurred dizem quantas áreas foram borradas.",
        },
        400: {"description": "O arquivo não é uma imagem válida"},
        401: {"description": "Chave ausente ou inválida no header X-API-Key"},
        413: {"description": "Imagem maior que 10 MB"},
        415: {"description": "Tipo de arquivo não aceito"},
        503: {"description": "Anonimização indisponível: nada foi devolvido"},
    },
)
def anonymize(
    file: UploadFile = File(description="Foto do cidadão (JPEG ou PNG)"),
    anonymizer: Anonymizer = Depends(anonymizer_dependency),
) -> Response:
    data = read_upload(file)

    try:
        image = decode_image(data)
    except InvalidImageError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error

    try:
        result = anonymizer.anonymize(image)
    except AnonymizationError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error

    return Response(
        content=encode_jpeg(result.image),
        media_type="image/jpeg",
        headers={
            "X-Faces-Blurred": str(len(result.faces)),
            "X-Plates-Blurred": str(len(result.plates)),
            "X-Persons-Blurred": str(len(result.persons)),
        },
    )
