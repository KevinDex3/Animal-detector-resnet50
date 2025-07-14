import cgi
import json
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

# Carica le etichette ImageNet
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS = requests.get(LABELS_URL).text.strip().split("\n")

def handle(event, context):
    try:
        # Rimuove i pesi ResNet50 dalla cache per forzare il download
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        for file_path in cache_dir.glob("resnet50-*.pth"):
            try:
                file_path.unlink()
            except Exception:
                pass  # Ignora eventuali errori di rimozione

        # Carica i pesi e la trasformazione per ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        model.eval()
        transform = weights.transforms()

        body = event.body
        if isinstance(body, str):
            body = body.encode('utf-8')

        headers = dict(event.headers)
        content_type = headers.get("content-type") or headers.get("Content-Type", "")
        content_length = str(len(body))

        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": content_length
        }

        fs = cgi.FieldStorage(
            fp=BytesIO(body),
            environ=environ,
            keep_blank_values=True
        )

        if 'file' not in fs:
            return {
                "statusCode": 400,
                "body": "Errore: nessun file trovato nella richiesta."
            }

        file_item = fs['file']
        image = Image.open(file_item.file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top3_probs, top3_classes = torch.topk(probabilities, 3)

        top3_results = []
        for i in range(3):
            class_idx = top3_classes[0][i].item()
            class_name = LABELS[class_idx]
            prob = top3_probs[0][i].item()
            top3_results.append({"class": class_name, "probability": prob})

        return {
            "statusCode": 200,
            "body": json.dumps({"top_3_predictions": top3_results}),
            "headers": {"Content-type": "application/json"}
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Errore durante l'elaborazione: {str(e)}"
        }
