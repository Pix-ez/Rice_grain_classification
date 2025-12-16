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

import os

from rice_dataset import RiceGrainDataset 
from helper import train_transforms, val_transforms  , get_modified_model
import warnings
warnings.filterwarnings('ignore')

import executorch.extension.pybindings.portable_lib
import executorch.kernels.quantized
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


full_dataset = RiceGrainDataset(root_dir='./rice-image-dataset/Rice_Image_Dataset', transform=train_transforms)



#doing 80 20 split for train test data
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

val_subset.dataset.transform = val_transforms

train_loader_batch_size_1 = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=2)
val_loader_batch_size_1 = DataLoader(val_subset, batch_size=1, shuffle=True, num_workers=2)

model = get_modified_model(num_classes=5)
#put correct base model name here
model.load_state_dict(torch.load("rice_model__epoch_0_loss_0.01.pth", map_location="cpu"))
model.eval()

# Create a dummy input for the export tracer
example_input = (torch.randn(1, 3, 224, 224),)

#Capturing the graph using torch.export
print("Exporting model graph...")
exported_model = export(model, example_input)

quantizer = XNNPACKQuantizer()

global_config = get_symmetric_quantization_config(is_per_channel=True)
quantizer.set_global(global_config)

linear_config = get_symmetric_quantization_config(is_per_channel=True)
quantizer.set_module_type(nn.Linear, linear_config)


print("Preparing model for quantization...")
prepared_model = prepare_pt2e(exported_model.module(), quantizer)

# calibrating on train data
print("Calibrating ...")

try:
#
    print("found train loader batch 1")
    data_iter = iter(train_loader_batch_size_1)
    # trying with different steps to see its effect
    for i in tqdm(range(1000)):
        images, _ = next(data_iter)
        # The prepared_model (now a GraphModule) expects a single tensor input
        prepared_model(images)
except Exception as e:
    print(f"Warning: 'train_loader' not found. Using random noise for calibration (NOT RECOMMENDED for accuracy). {e}")

print("Converting to Int8...")
# conveting to int8
quantized_model_module = convert_pt2e(prepared_model)

print("Re-exporting quantized graph...")

quantized_exported_program = torch.export.export(quantized_model_module, example_input)

print("Lowering to ExecuTorch Edge IR...")

edge_program_manager = to_edge_transform_and_lower(
    quantized_exported_program,
    partitioner=[XnnpackPartitioner()]
)

executorch_program = edge_program_manager.to_executorch()
# Save the file
output_file = "rice_model_quantized.pte"
with open(output_file, "wb") as f:
    f.write(executorch_program.buffer)

print(f"SUCCESS! Model saved to {output_file}")
file_size = os.path.getsize(output_file) / (1024 * 1024)

print(f"✅ File size: {file_size:.2f} MB")