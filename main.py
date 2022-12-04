from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from monai.networks.nets import DenseNet121
from monai.data import Dataset, DataLoader
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    ScaleIntensity,
    ToTensor,
)
import torch
from uvicorn import run
import os
import requests

class MedNISTDataset(Dataset):

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

app = FastAPI()

class_predictions = [
    'AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT'
]

num_class = len(class_predictions)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet121(
    spatial_dims=2,
    in_channels=1,
    out_channels=num_class
).to(device)
model_dir = 'best_metric_model.pth'
model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))


# @app.get("/")
# async def root():
#     return {"message": "Welcome to the medical image classification API!"}

@app.post("/predict/")
async def get_net_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}
    
    val_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        ToTensor()
    ])

    response = requests.get(image_link)
    if response.status_code:
        fp = open(image_link.split('/')[-1], 'wb')
        fp.write(response.content)
        fp.close()
    test_ds = MedNISTDataset([image_link.split('/')[-1]], list(range(len([image_link]))), val_transforms)
    test_loader = DataLoader(test_ds, batch_size=300, num_workers=2)

    for test_data in test_loader:
        test_images = test_data[0].to(device)
        pred = model(test_images).argmax(dim=1)

    class_prediction = class_predictions[pred[0].item()]


    return {
        "model-prediction": class_prediction,
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)
