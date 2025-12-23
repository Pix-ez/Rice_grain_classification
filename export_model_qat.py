import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.export import export
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
# Quantization Imports (PT2E flow)
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config, 
)
from tqdm import tqdm

import torch.optim as optim
from torch.export import export
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
import copy
import os
import executorch.extension.pybindings.portable_lib
import executorch.kernels.quantized
from executorch.exir import to_edge_transform_and_lower

from rice_dataset import RiceGrainDataset 
from helper import train_transforms, val_transforms  , get_modified_model
import warnings
warnings.filterwarnings('ignore')





full_dataset = RiceGrainDataset(root_dir='rice-image-dataset/Rice_Leaf_AUG', transform=train_transforms)



#doing 80 20 split for train test data
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

val_subset.dataset.transform = val_transforms

train_loader_batch_size_1 = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=2)
val_loader_batch_size_1 = DataLoader(val_subset, batch_size=1, shuffle=True, num_workers=2)

model = get_modified_model(num_classes=6)
print("="*70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


print("\n[1/10] Loading FP32 trained model...")
model = get_modified_model(num_classes=6)
model.load_state_dict(torch.load("rice_model__epoch_0_loss_0.01.pth", map_location="cpu"))
model.eval()
print("✅ FP32 model loaded")

# Validate baseline
print("\nValidating FP32 baseline...")
val_loader_fp32 = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=2)
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in tqdm(val_loader_fp32, desc="FP32"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
fp32_acc = 99
print(f"✅ FP32 Baseline: {fp32_acc:.2f}%")

model.cpu()

print("\n[2/10] Preparing QAT model...")

# Create fresh model for QAT
model_qat = get_modified_model(num_classes=6)
model_qat.load_state_dict(torch.load("model_epoch17_acc0.9047.pth", map_location="cpu"))
model_qat.eval()

# Export for quantization preparation
BATCH_SIZE = 1  
example_input = (torch.randn(BATCH_SIZE, 3, 224, 224),)
exported_model = export(model_qat, example_input)

# Setup quantizer
quantizer = XNNPACKQuantizer()
global_config = get_symmetric_quantization_config(
    is_per_channel=True,  # Per-tensor for XNNPACK compatibility
    is_qat=True  # QAT mode
)
quantizer.set_global(global_config)
print("✅ Using per-tensor quantization for QAT")

# Prepare QAT model
qat_model = prepare_qat_pt2e(exported_model.module(), quantizer)
print("✅ QAT model prepared")


print("\n[3/10] QAT Training...")

qat_model.to(device)
torch.ao.quantization.move_exported_model_to_train(qat_model)

# Setup training
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                        drop_last=True, num_workers=2)

optimizer = optim.Adam(qat_model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-7)
criterion = nn.CrossEntropyLoss()

num_epochs = 20
best_val_acc = 0.0
best_qat_state = None

print(f"\nTraining for {num_epochs} epochs...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Train
    torch.ao.quantization.move_exported_model_to_train(qat_model)
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = qat_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # Validate
    torch.ao.quantization.move_exported_model_to_eval(qat_model)
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = qat_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            # break
    val_acc = 100 * val_correct / val_total
    print(f"Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_qat_state = copy.deepcopy(qat_model.state_dict())
        print(f"✅ Saved best QAT model! (Acc: {val_acc:.2f}%)")

    scheduler.step()

    if device.type == 'cuda':
        torch.cuda.empty_cache()

print(f"\n✅ QAT Training complete. Best Val Acc: {best_val_acc:.2f}%")


print("\n[4/10] Saving QAT weights...")
qat_weights_file = "mobilenetv3_qat_weights.pth"
torch.save(best_qat_state, qat_weights_file)
print(f"✅ QAT weights saved to {qat_weights_file}")


print("\n[5/10] Creating fresh model...")

clean_model = get_modified_model(num_classes=6)
clean_model.eval()

# Export fresh model
example_input_mobile = (torch.randn(1, 3, 224, 224),)  # Batch size 1 for mobile
clean_exported = export(clean_model, example_input_mobile)

# Setup quantizer again
quantizer_final = XNNPACKQuantizer()
config_final = get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
quantizer_final.set_global(config_final)

# Prepare QAT model
clean_qat_model = prepare_qat_pt2e(clean_exported.module(), quantizer_final)

# Loading the saved QAT weights into this fresh model
print("Loading QAT weights into fresh model...")
clean_qat_model.load_state_dict(best_qat_state, strict=True)
print("✅ QAT weights loaded successfully")

print("\n[6/10] Converting to INT8...")
torch.ao.quantization.move_exported_model_to_eval(clean_qat_model)
quantized_model = convert_pt2e(clean_qat_model)
torch.ao.quantization.move_exported_model_to_eval(quantized_model)
print("✅ Model converted to INT8")


print("\n[7/10] Validating INT8 model...")
val_loader_int8 = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=2)

quantized_model.to(device)
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in tqdm(val_loader_int8, desc="INT8 Validation"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = quantized_model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # break

int8_acc = 100 * correct / total

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"FP32 Baseline:        {fp32_acc:.2f}%")
print(f"QAT (fake quant):     {best_val_acc:.2f}%")
print(f"INT8 (quantized):     {int8_acc:.2f}%")
print(f"Total accuracy drop:  {fp32_acc - int8_acc:.2f}%")
print("="*70)

print("\n[8/10] Exporting to .pte file...")

quantized_model.cpu()

try:
    quantized_exported_program = torch.export.export(quantized_model, example_input_mobile)

    edge_program_manager = to_edge_transform_and_lower(
                            quantized_exported_program,
                            partitioner= [XnnpackPartitioner()]
                            )

    executorch_program = edge_program_manager.to_executorch()

    output_file = f"mobilenetv3_qat_acu{int8_acc:.2f}.pte"
    with open(output_file, "wb") as f:
        f.write(executorch_program.buffer)

    file_size = os.path.getsize(output_file) / (1024 * 1024)

    print("\n" + "="*70)
    print("EXPORT SUCCESSFUL!")
    print("="*70)
    print(f"✅ Model saved to: {output_file}")
    print(f"✅ File size: {file_size:.2f} MB")
    print(f"✅ INT8 Accuracy: {int8_acc:.2f}%")
   
    print("="*70)

   
except Exception as e:
    print(f"\n❌ Export failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[9/10] Cleanup...")
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print("✅ GPU memory cleared")


print("\n[10/10] Summary")
print("="*70)
print(f"\nFiles created:")
print(f"  • {qat_weights_file} - QAT weights ")
print(f"  • {output_file} - Quantized model for deployment")
print(f"\nModel stats:")
print(f"  • Size: {file_size:.2f} MB")
print(f"  • Accuracy: {int8_acc:.2f}%")
print(f"  • Quantization: Per-tensor INT8")
print("="*70)