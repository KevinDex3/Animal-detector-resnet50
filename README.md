# 🐾 Animal Detector (ResNet50)

Questa funzione serverless utilizza **ResNet50**, un modello di deep learning pre-addestrato su **ImageNet**, per classificare immagini e restituire le **prime 3 predizioni** con le rispettive probabilità.

---

## 📦 Funzionalità

- Accetta una richiesta `POST` `multipart/form-data` contenente un'immagine (`file`)
- Usa `torchvision.models.resnet50` con i pesi `IMAGENET1K_V1`
- Restituisce le 3 classi più probabili con probabilità associate
- Compatibile con ambienti serverless (es. **AWS Lambda**, **OpenFaaS**, ecc.)

---

## 🧠 Esempio di risposta JSON

```json
{
  "top_3_predictions": [
    {"class": "golden retriever", "probability": 0.87},
    {"class": "Labrador retriever", "probability": 0.08},
    {"class": "cocker spaniel", "probability": 0.03}
  ]
}

```
## ⌨️​ Esempio di input

curl -X POST http://INDIRIZZO/function/animal-detector-resnet50  -H "Content-Type: multipart/form-data"   -F "file=@/home/kevin/Immagini/cane.jpeg"
Occorre prima salvare un'immagine e successivamente specificare il percorso appropriato!
