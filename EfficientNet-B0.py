import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import os

class DeepfakeDetectionModel(nn.Module):
    def __init__(self):
        super(DeepfakeDetectionModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2)  # 2 classes: real or fake
        )
        
    def forward(self, x):
        return self.backbone(x)

class FaceForensicsDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_video=1):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.face_detector = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.video_paths = []
        self.labels = []

        # Load videos from 'real' and 'fake' subfolders
        for label in ['real', 'fake']:
            label_dir = os.path.join(root_dir, label)
            for video_file in os.listdir(label_dir):
                self.video_paths.append(os.path.join(label_dir, video_file))
                self.labels.append(0 if label == 'real' else 1)

    def __len__(self):
        return len(self.video_paths)
    
    def extract_faces(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.face_detector.detect(frame_rgb)
        
        if boxes is None or len(boxes) == 0:
            return None
        
        # Get the largest face
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_face_idx = areas.argmax()
        box = boxes[largest_face_idx]
        
        face = frame_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        face = Image.fromarray(face)
        
        if self.transform:
            face = self.transform(face)
            
        return face
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        random_frame = np.random.randint(0, total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Fallback to zero tensor if frame read fails
            return torch.zeros(3, 224, 224), self.labels[idx]
            
        face = self.extract_faces(frame)
        if face is None:
            # Fallback to zero tensor if no face detected
            return torch.zeros(3, 224, 224), self.labels[idx]
            
        return face, self.labels[idx]

def train_model(model, train_loader, val_loader, num_epochs=10, save_path="deepfake_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_accuracy = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_accuracy = 100. * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Acc: {val_accuracy:.2f}%")

def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    test_accuracy = 100. * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    return test_accuracy

def predict_video(model, video_path, transform, frame_sample_rate=5, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
    
    face_detector = MTCNN(keep_all=True, device=device)
    results = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_sample_rate != 0:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = face_detector.detect(frame_rgb)
        
        if boxes is None or len(boxes) == 0:
            continue
            
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_face_idx = areas.argmax()
        box = boxes[largest_face_idx]
        
        face = frame_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        face = Image.fromarray(face)
        face = transform(face).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(face)
            probs = torch.softmax(output, dim=1)
            prob_fake = probs[0][1].item()
            
        results.append(prob_fake)
    
    cap.release()
    
    if not results:
        print(f"No faces detected in video: {video_path}")
        return None
    
    avg_prob_fake = np.mean(results)
    prediction = "Fake" if avg_prob_fake > threshold else "Real"
    
    return {
        'video_path': video_path,
        'avg_probability_fake': avg_prob_fake,
        'prediction': prediction
    }

def main():
    # Data preprocessing with augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset paths (update with your local path)
    dataset_root = "path/to/1000-videos-split"  # Replace with actual path
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "validation")
    test_dir = os.path.join(dataset_root, "test")
    
    # Create datasets
    train_dataset = FaceForensicsDataset(train_dir, transform=transform)
    val_dataset = FaceForensicsDataset(val_dir, transform=transform)
    test_dataset = FaceForensicsDataset(test_dir, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = DeepfakeDetectionModel()
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=10, save_path="deepfake_efficientnet.pth")
    
    # Load best model and test
    model.load_state_dict(torch.load("deepfake_efficientnet.pth", map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    test_model(model, test_loader)
    
    # Example video inference
    sample_video = os.path.join(test_dir, "fake", "fake_video1.mp4")  # Replace with actual video name
    result = predict_video(model, sample_video, transform)
    if result:
        print(f"Video: {result['video_path']}")
        print(f"Prediction: {result['prediction']}, Avg Fake Prob: {result['avg_probability_fake']:.2f}")

if __name__ == "__main__":
    main()
