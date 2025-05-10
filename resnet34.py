# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

#%%
import os
import random
from tqdm import tqdm

SOURCE_DIR = "dataset_imagess"
DEST_DIR = "splited_dataset"
TRAIN_RATIO = 0.8
NUM_CLASSES = 500
SEED = 42

all_classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d)) ]
random.seed(SEED)
selected_classes = random.sample(all_classes, NUM_CLASSES)


# %%
for split in ["train", "val"]:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)


# %%
import shutil
for class_name in tqdm(selected_classes, desc="Processing classes"):
    src_class_dir = os.path.join(SOURCE_DIR, class_name)
    images = [f for f in os.listdir(src_class_dir) if os.path.isfile(os.path.join(src_class_dir, f))]
    random.shuffle(images)

    train_count = int(len(images) * TRAIN_RATIO)
    train_images = images[:train_count]
    val_images = images[train_count:]

    for split_name, split_images in zip(["train", "val"], [train_images, val_images]):
        dst_class_dir = os.path.join(DEST_DIR, split_name, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        for img in split_images:
            shutil.copy(os.path.join(src_class_dir, img), os.path.join(dst_class_dir, img))




#  %%
# Config
NUM_CLASSES = 500
BATCH_SIZE = 64
image_size = 224
EPOCHS = 5
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRAIN_DIR = "C:\\Users\\MSR\\fonts_dataset\\train"
VAL_DIR = "C:\\Users\\MSR\\fonts_dataset\\val"
SAVE_PATH = "resnet_model9.pth"
# %%


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        logprobs = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# %%

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])


val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transform)


# 5. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%



model = models.resnet34(pretrained=True).to(DEVICE)

# %%
# model.load_state_dict(torch.load("resnet_model1.pth"))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# %%

best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print("Model saved.")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break
# %%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# Configuration
NUM_CLASSES = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "resnet_model9.pth" 
IMAGE_SIZE = 224 

model = models.resnet34(pretrained=False)  # or whatever ResNet version you used
model = model.to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
def predict_image_topk(image_path, topk=15):
    img = Image.open(image_path).convert('RGB')
    img = train_transform(img)
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)  
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=topk)

    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    results = []
    for i in range(1):
        class_index = top_indices[i]
        class_name = train_dataset.classes[class_index]
        confidence = top_probs[i]
        results.append((class_name, confidence))

    return results

image_path ="C:\\Users\\MSR\\fonts_dataset\\train\\Al Nile\\0.png"
top5_results = predict_image_topk(image_path,3)

print("Top 5 Predictions:")
for font, score in top5_results:
    print(f"{font}: {score:.4f}")
# %%
print(train_dataset.classes)
# %%
