import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ---------------- MODEL DEFINITION ----------------
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(
            ConvBlock(128,128),
            ConvBlock(128,128)
        )
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(
            ConvBlock(512,512),
            ConvBlock(512,512)
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)

# ---------------- LOAD CLASSES (ORDER IS CRITICAL) ----------------
with open("ai/classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# ---------------- LOAD FULL MODEL ----------------
model = torch.load(
    "ai/plant-disease-model.pth",
    map_location="cpu",
    weights_only=False   # trusted local model
)

model.eval()

# ---------------- TRANSFORM (MATCH TRAINING DISTRIBUTION) ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- PREDICT ----------------
img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)
    confidence, idx = torch.max(probs, dim=1)

# Print ONLY class name (server parses this)
print(classes[idx.item()])
print(f"Confidence: {confidence.item()*100:.2f}%")