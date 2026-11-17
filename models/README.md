# Modelos

| Pasta | Modelo | Para quê | Tamanho | Licença | Origem | No Git? |
|---|---|---|---|---|---|---|
| `anonymizer/` | `face_detection_yunet_2023mar.onnx` | Rostos (borrar) | 227 KB | MIT | [OpenCV Zoo · YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Sim |
| `anonymizer/` | `yolo-v9-t-640-license-plates-end2end.onnx` | Placas (borrar) | 7,5 MB | MIT | [open-image-models](https://github.com/ankandrew/open-image-models/releases/tag/assets) | Sim |
| `anonymizer/` | `rf-detr-nano-384-coco.onnx` | Pessoas de longe (borrar só a cabeça) | 114 MB | Apache 2.0 | [open-image-models](https://github.com/ankandrew/open-image-models/releases/tag/assets) | **Não**: passa de 100 MB. `python -m scripts.baixar_modelos` |
| `crosswalk/` | modelo próprio da faixa | Faixa nítida × apagada | ~5 MB | — | treinado por nós | Sim (próxima fase) |
| `pothole/` | `smartathon-yolov8s.onnx` (escolhido) | Buraco | 43 MB | o card não declara | [Samdutse/pothole-yolov8](https://huggingface.co/Samdutse/pothole-yolov8), convertido de `.pt` para ONNX | Não (ver abaixo) |
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

## SHA-256 (conferência de integridade)

O hash é como o lacre de um pacote: se o arquivo mudar um único byte, o hash muda.

```
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  anonymizer/face_detection_yunet_2023mar.onnx
c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076  anonymizer/yolo-v9-t-640-license-plates-end2end.onnx
fdcdd1c566ed4ab4d3700ca1ce6f03edadfec2a0655432e40e4c2734623feaec  anonymizer/rf-detr-nano-384-coco.onnx
```

Sem o `rf-detr-nano-384-coco.onnx`, a API sobe, mas o `/health` mostra `anonymizer: unavailable`
e o `/analisar` responde 503 (falha segura: nenhuma imagem é processada sem o borrão completo).

Conferir: `sha256sum models/anonymizer/*.onnx`
