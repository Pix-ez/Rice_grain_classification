import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader
from executorch.runtime import Runtime
from helper import  train_transforms, val_transforms
from rice_dataset import RiceGrainDataset
from torch.utils.data import  DataLoader, random_split

from executorch.runtime import Runtime, Program, Method
from helper import plot_confusion_matrix

full_dataset = RiceGrainDataset(root_dir='./rice-image-dataset/Rice_Image_Dataset', transform=train_transforms)

print(f"Total Images Found: {len(full_dataset)}")
print(f"Class Mapping: {full_dataset.class_to_idx}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

val_subset.dataset.transform = val_transforms



eval_loader = DataLoader(val_subset, batch_size=1, shuffle=True, drop_last=True)

print(f"Total Test Images: {len(eval_loader)}")
#set correct path for model and name
pte_path = "models/rice_model_ptq_optimized_99.pte"
print(f"Loading {pte_path}...")

try:
    runtime_instance = Runtime.get()
    program = runtime_instance.load_program(pte_path)
    method_name = "forward"
    method = program.load_method(method_name)
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")

y_true = []
y_pred = []
inference_times = []

print("Starting Inference on Validation Set...")


for inputs, labels in tqdm(eval_loader, desc="Evaluating"):
    outputs = method.execute([inputs])
    logits = outputs[0]
    predicted_id = torch.argmax(logits, dim=1).item()

    y_pred.append(predicted_id)
    y_true.append(labels.item())


print("\n--- Evaluation Results ---")
print(f"{pte_path}")
acc = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {acc * 100:.2f}%")

class_names = [full_dataset.idx_to_class[i] for i in range(len(full_dataset.class_to_idx))]

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nDetailed Classification Report:")
print(report)

plot_confusion_matrix(y_true, y_pred, class_names)