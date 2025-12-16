import torch
import torch.nn as nn
from torchvision import models
import time
import copy
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import  confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#



# for mobilenet
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize to standard input
    transforms.RandomHorizontalFlip(),  # Augmentation: Flip left/right
    transforms.RandomRotation(15),      # Augmentation: Slight rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Handle lighting variations
    transforms.ToTensor(),              # Convert [0, 255] to Tensor [0.0, 1.0]
    transforms.Normalize(norm_mean, norm_std)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])



def get_modified_model(num_classes=5):
    # Loading Pre-trained MobileNetV3 Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    #changing last layer from conv feature block to linear layer for our classification task
    in_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model





def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    since = time.time()

    # Store results to plot later
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        model.train()  
        running_loss = 0.0
        running_corrects = 0

        for i ,(inputs, labels ) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()   
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(val_loader)):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        print(f'Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"rice_model__epoch:{epoch+1}_loss:{val_loss:.2f}.pth")
            print("  -> Best model saved!")

        print("\n")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Original Float32 Model)')
    plt.show()