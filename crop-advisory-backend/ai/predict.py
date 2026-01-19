import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ================= MODEL =================
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
    def __init__(self, in_channels=3, num_classes=38):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
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

# ================= LOAD CLASSES =================
with open("ai/classes.txt") as f:
    classes = [c.strip() for c in f.readlines()]

# ================= LOAD MODEL =================
device = torch.device("cpu")

model = torch.load(
    "ai/plant-disease-model.pth",
    map_location=device,
    weights_only=False
)

model.eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= PREDICT =================
img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(img)
    probs = torch.softmax(logits, dim=1)[0]

# ================= HARD SAFETY RULE =================
top1_idx = torch.argmax(probs).item()
top1_class = classes[top1_idx]
top1_conf = probs[top1_idx].item()

# find tomato healthy prob
tomato_healthy_prob = 0.0
if "Tomato___healthy" in classes:
    tomato_healthy_prob = probs[classes.index("Tomato___healthy")].item()

# 🚨 FINAL DECISION
if (
    "Powdery_mildew" in top1_class
    and tomato_healthy_prob >= 0.30
):
    print("Tomato___healthy")
    print(f"Confidence: {tomato_healthy_prob*100:.2f}%")
else:
    print(top1_class)
    print(f"Confidence: {top1_conf*100:.2f}%")
