from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
from torchvision.models import mobilenet_v2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Define the model with MobileNetV2 backbone
def get_model(num_classes):
    backbone = mobilenet_v2(weights="IMAGENET1K_V1").features
    backbone.out_channels = 1280

    # Define the AnchorGenerator
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

# Load the trained model
def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Run inference, extract feature maps, and save the image with bounding boxes
def extract_feature_maps_and_boxes(image_path, model, device, output_dir, class_names, threshold=0.5):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    feature_maps = {}

    # Hook to extract the backbone output
    def hook_fn(module, input, output):
        feature_maps["backbone"] = output

    # Register the hook
    hook = model.backbone.register_forward_hook(hook_fn)

    # Perform inference to trigger the hook and get model outputs
    with torch.no_grad():
        outputs = model(img_tensor)

    # Remove the hook after extraction
    hook.remove()

    # Get the bounding boxes, labels, and scores
    boxes = outputs[0]["boxes"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()

    # Filter results based on the threshold
    filtered_boxes = []
    filtered_labels = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            filtered_boxes.append(box)
            filtered_labels.append((class_names[label], score))

    # Save the image with bounding boxes
    save_image_with_boxes(image, filtered_boxes, filtered_labels, output_dir)

    # Save the first 5 feature maps with heatmap colors
    feature_map_tensor = feature_maps["backbone"]
    save_feature_maps_with_heatmap(feature_map_tensor, image.size, output_dir)

# Save the original image with bounding boxes and labels
def save_image_with_boxes(image, boxes, labels, output_dir):
    # Make a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    for box, (label, score) in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=15)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the image with bounding boxes and labels
    image_path = os.path.join(output_dir, "image_with_boxes.png")
    draw_image.save(image_path)
    print(f"Image with bounding boxes saved to {image_path}")

# Save the feature maps as images with heatmap colors
def save_feature_maps_with_heatmap(feature_map_tensor, image_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(min(5, feature_map_tensor.shape[1])):
        fmap = feature_map_tensor[0, i].cpu().detach().numpy()
        
        # Normalize the feature map for better contrast in visualization
        fmap = (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap))

        # Use matplotlib to create a heatmap and save it as an image
        plt.figure(figsize=(6, 6))
        plt.imshow(fmap, cmap="viridis", interpolation="bilinear")
        plt.axis("off")

        # Save the heatmap to file
        feature_map_path = os.path.join(output_dir, f"feature_map_channel_{i+1}.png")
        plt.savefig(feature_map_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Feature map {i+1} saved to {feature_map_path}")

# Main function
if __name__ == "__main__":
    # Model parameters
    model_path = "faster_rcnn_cat_dog.pth"
    num_classes = 3
    class_names = ["background", "cat", "dog"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model
    model = load_model(model_path, num_classes)
    model.to(device)

    # Image path and output directory
    image_path = "images/Jul/Easiest.jpg"
    output_dir = "images/Output"

    # Extract feature maps and save image with bounding boxes
    extract_feature_maps_and_boxes(image_path, model, device, output_dir, class_names)
