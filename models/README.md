# Modelos

| Pasta | Modelo | Para quê | Tamanho | Licença | Origem | No Git? |
|---|---|---|---|---|---|---|
| `anonymizer/` | `face_detection_yunet_2023mar.onnx` | Rostos (borrar) | 227 KB | MIT | [OpenCV Zoo · YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Sim |
| `anonymizer/` | `yolo-v9-t-640-license-plates-end2end.onnx` | Placas (borrar) | 7,5 MB | MIT | [open-image-models](https://github.com/ankandrew/open-image-models/releases/tag/assets) | Sim |
| `anonymizer/` | `rf-detr-nano-384-coco.onnx` | Pessoas de longe (borrar só a cabeça) | 114 MB | Apache 2.0 | [open-image-models](https://github.com/ankandrew/open-image-models/releases/tag/assets) | **Não**: passa de 100 MB. `python -m scripts.baixar_modelos` |
| `crosswalk/` | modelo próprio da faixa | Faixa nítida × apagada | ~5 MB | — | treinado por nós | Sim (próxima fase) |
| `pothole/` | `smartathon-yolov8s.onnx` (escolhido) | Buraco | 43 MB | o card não declara | [Samdutse/pothole-yolov8](https://huggingface.co/Samdutse/pothole-yolov8), convertido de `.pt` para ONNX | **Não**: o autor só publica o `.pt`. Convertido no build (ver abaixo) |
| `pothole/` | `rdd-yolov8s.onnx` (descartado) | Buraco | 45 MB | Apache 2.0 | [vinothvikas1987/pothole-detection-yolov8](https://huggingface.co/vinothvikas1987/pothole-detection-yolov8) | Não |

## Buraco: por que o modelo B

Comparados em 100 fotos com buraco (dataset "Potholes Detection", Roboflow, CC BY 4.0) + 6 fotos de rua
sem buraco, com `python -m scripts.avaliar_buraco` (IoU ≥ 0,5, limiar 0,25):

| Modelo | Precisão | Recall | F1 |
|---|---|---|---|
| A · RDD (Japão/Índia, 5 classes) | 42,9% | 1,1% | 2,2% |
| B · Smartathon (Arábia Saudita) | 82,0% | 74,3% | 78,0% |

O A chama quase todo buraco desse conjunto de "Other". Em câmera de rua, o B confunde sombra, tampa de
bueiro, faixa pintada e moto: por isso as caixas passam por zonas da câmera e por um verificador LLM
(`app/vision/pothole_check.py`).

Com o verificador LLM, no mesmo conjunto (`python -m scripts.avaliar_buraco --verificar`):

| Versão do verificador | Precisão | Recall | F1 | Fotos sem buraco com alarme |
|---|---|---|---|---|
| 1ª (rígida: poça e remendo nunca eram buraco) | 87,7% | 39,8% | 54,7% | 0 de 6 |
| 2ª (buraco com água e remendo afundado contam) | 85,7% | 60,2% | 70,7% | 0 de 6 |

O recall fica abaixo do modelo sozinho porque o LLM recusa parte dos buracos reais e, no plano
gratuito, 25 caixas ficaram sem resposta (limite de chamadas): sem segunda opinião, a caixa fica
abaixo de 0,5 e não abre ocorrência. A troca é consciente: menos ocorrências falsas (sombra, bueiro).

## Buraco: conversão no build

O autor publica só o `.pt` (PyTorch). O serviço roda com ONNX Runtime, sem PyTorch, então o
Dockerfile tem um estágio só para converter (`scripts/exportar_buraco.py` + `requirements-export.txt`):

1. baixa o `best.pt` e confere o SHA-256 (se mudar na origem, o build para);
2. converte com o Ultralytics (640×640, opset 12) e confere a saída do ONNX: `(1, 5, 8400)`;
3. só o `.onnx` passa para a imagem final: o PyTorch fica no estágio descartado.

Local (fora do Docker), num venv separado: `pip install -r requirements-export.txt` e
`python scripts/exportar_buraco.py models/pothole/smartathon-yolov8s.onnx`.
Conferido: o ONNX do build dá as mesmas caixas e confianças do usado na avaliação (30 fotos, 55 caixas).

Melhoria futura: publicar o `.onnx` convertido como release do repositório, com SHA-256 fixo,
e baixar como os outros modelos (build mais rápido, sem PyTorch em nenhum estágio).

## SHA-256 (conferência de integridade)

O hash é como o lacre de um pacote: se o arquivo mudar um único byte, o hash muda.

```
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  anonymizer/face_detection_yunet_2023mar.onnx
c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076  anonymizer/yolo-v9-t-640-license-plates-end2end.onnx
fdcdd1c566ed4ab4d3700ca1ce6f03edadfec2a0655432e40e4c2734623feaec  anonymizer/rf-detr-nano-384-coco.onnx
1067a09d37e4e9081164e6d6da337ad66d8ec7a884a6a194e58703beee804aa2  best.pt do Samdutse (antes de converter)
```

Sem o `rf-detr-nano-384-coco.onnx`, a API sobe, mas o `/health` mostra `anonymizer: unavailable`
e o `/analisar` responde 503 (falha segura: nenhuma imagem é processada sem o borrão completo).

Conferir: `sha256sum models/anonymizer/*.onnx`
