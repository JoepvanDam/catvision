from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import mobilenet_v2
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
import torch
import os

scaler = GradScaler()

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annotations", self.annots[idx])
        img = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_voc_xml(annot_path)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        # Convert image to tensor
        img = F.to_tensor(img)
        return img, target


    def parse_voc_xml(self, annot_path):
        tree = ET.parse(annot_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            # Map label to class index (cat=1, dog=2)
            labels.append(1 if label == "cat" else 2)
        return boxes, labels

    def __len__(self):
        return len(self.imgs)

# Define Model
def get_model(num_classes):
    # Load the MobileNet v2 model
    backbone = mobilenet_v2(weights="IMAGENET1K_V1").features
    backbone.out_channels = 1280 
    
    # Define an AnchorGenerator for a single feature map
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )
    return model

# Training Code
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += losses.item()
    return running_loss / len(data_loader)

# Main Script
def main():
    dataset_path = r"/mnt/c/Programming/Web/Kennisdeling-CV/catvision/dataset"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize dataset and data loaders
    dataset = CustomDataset(dataset_path, transforms=None)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=6, pin_memory=True)

    # Load model
    num_classes = 3
    model = get_model(num_classes)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "faster_rcnn_cat_dog.pth")
    print("Model training complete!")

if __name__ == "__main__":
    main()
