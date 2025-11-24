import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

device = "cpu"

model = SimpleCNN()
try:
    model.load_state_dict(torch.load("cats_vs_dogs_cnn.pth", map_location=device))
    model.eval()
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

app = FastAPI()

class PredictionResponse(BaseModel):
    filename: str
    prediction: str

@app.get("/")
async def root():
    return {"message": "Servicio de inferencia de gatos vs perros"}

@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Endpoint para subir una imagen y obtener la predicción."""
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
            
        prediction = output.item()
        
        if prediction < 0.5:
            label = "Gato"
        else:
            label = "Perro"
            
        return {"filename": file.filename, "prediction": label}

    except Exception as e:
        return {"filename": file.filename, "prediction": f"Error: {e}"}