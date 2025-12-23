import torch
import time
import psutil
import platform
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from helper import val_transforms, train_transforms
from rice_dataset import RiceGrainDataset
from torch.utils.data import Dataset, DataLoader, random_split

full_dataset = RiceGrainDataset(root_dir='rice-image-dataset/Rice_Leaf_AUG', transform=train_transforms)
# #using batch_size = 64 we can adjust this on training memory resource
BATCH_SIZE = 64

print(f"Total Images Found: {len(full_dataset)}")
print(f"Class Mapping: {full_dataset.class_to_idx}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

val_subset.dataset.transform = val_transforms

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,  num_workers=2)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,  drop_last=True, num_workers=2)

train_loader_batch_size_1 = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=2)
val_loader_batch_size_1 = DataLoader(val_subset, batch_size=1, shuffle=True, num_workers=2)

# =============================================================================
# WRAPPER CLASS FOR EXECUTORCH
# =============================================================================
class ExecuTorchWrapper:
    """
    A wrapper to make the .pte model behave like a PyTorch model
    so we can use the same evaluation loop.
    """
    def __init__(self, pte_path):
        self.pte_path = pte_path
        print(f"Loading ExecuTorch model from: {pte_path}")
        # Load the module using the portable runtime
        self.module = _load_for_executorch(pte_path)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # ExecuTorch expects inputs as a list/tuple
        # We ensure input is on CPU because the Python runtime usually runs on CPU
        inputs = [x.cpu()]
        
        # Run inference
        result = self.module.forward(inputs)
        
        # result is a list of EValues. Assuming single output tensor at index 0.
        return result[0]


    def eval(self):
        # ExecuTorch is always in eval mode
        pass

# =============================================================================
# System Information
# =============================================================================

def print_system_info():
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print("="*70)

# =============================================================================
# Evaluation with Profiling
# =============================================================================

def evaluate_executorch_model(model_wrapper, val_loader, class_names, warmup_steps=5):
    """
    Evaluate ExecuTorch (.pte) model with performance profiling
    """
    
    print("\n" + "="*70)
    print("EXECUTORCH MODEL EVALUATION (.pte)")
    print("="*70)
    
    # Model info - Get physical file size since .parameters() doesn't exist
    file_size_mb = os.path.getsize(model_wrapper.pte_path) / (1024 * 1024)
    print(f"Model File Size: {file_size_mb:.2f} MB")
    
    # Initialize tracking variables
    inference_times = []
    memory_usage = []
    cpu_usage = []
    y_true = []
    y_pred = []
    
    process = psutil.Process()
    
    # =========================================================================
    # WARMUP PHASE
    # =========================================================================
    print(f"\nWarming up model ({warmup_steps} iterations)...")
    warmup_count = 0
    
    # No grad is not strictly necessary for ExecuTorch (it doesn't track grads), but good practice
    with torch.no_grad():
        for inputs, _ in val_loader:
            if warmup_count >= warmup_steps:
                break
            # Run inference
            _ = model_wrapper(inputs)
            warmup_count += 1
    
    print("Warmup complete!")
    
    # =========================================================================
    # EVALUATION PHASE
    # =========================================================================
    print("\nRunning evaluation with profiling...")
    
    # Standard PyTorch Dataloader loop
    for i, (inputs, labels) in enumerate(tqdm(val_loader, desc="Evaluating")):
        # Limit for testing if needed (remove this if you want full dataset)
        if i > 100: break 
        
        labels_np = labels.numpy()
        
        # Measure inference time
        start_time = time.perf_counter()
        
        # Forward pass (calls the wrapper)
        outputs = model_wrapper(inputs)
        
        end_time = time.perf_counter()
        batch_time = end_time - start_time
        
        # Get predictions
        # outputs is already a tensor from our wrapper
        _, preds = torch.max(outputs, 1)
        preds_np = preds.numpy()
        
        # Record metrics
        inference_times.append(batch_time)
        memory_usage.append(process.memory_info().rss / (1024 * 1024))  # MB
        cpu_usage.append(process.cpu_percent())
        
        # Store predictions and labels
        y_true.extend(labels_np)
        y_pred.extend(preds_np)
    
    # =========================================================================
    # CALCULATE STATISTICS
    # =========================================================================
    inference_times_ms = np.array(inference_times) * 1000  # Convert to milliseconds
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    # Timing stats
    print(f"\nInference Time:")
    print(f"  Average: {np.mean(inference_times_ms):.2f} ms")
    print(f"  Median:  {np.median(inference_times_ms):.2f} ms")
    print(f"  Std Dev: {np.std(inference_times_ms):.2f} ms")
    print(f"  Min:     {np.min(inference_times_ms):.2f} ms")
    print(f"  Max:     {np.max(inference_times_ms):.2f} ms")
    
    # Throughput
    total_samples = len(y_pred)
    total_time_s = np.sum(inference_times)
    throughput = total_samples / total_time_s
    print(f"\nThroughput: {throughput:.2f} images/second")
    print(f"Total Samples: {total_samples}")
    print(f"Total Time: {total_time_s:.2f} seconds")
    
    # Memory stats
    print(f"\nSystem Memory Usage:")
    print(f"  Average: {np.mean(memory_usage):.2f} MB")
    print(f"  Peak:    {np.max(memory_usage):.2f} MB")
    
    # CPU stats
    print(f"\nCPU Usage:")
    print(f"  Average: {np.mean(cpu_usage):.2f}%")
    print(f"  Peak:    {np.max(cpu_usage):.2f}%")
    
    # =========================================================================
    # ACCURACY METRICS
    # =========================================================================
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*70)
    print("ACCURACY METRICS")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    return {
        'accuracy': accuracy,
        'avg_inference_time_ms': np.mean(inference_times_ms),
        'throughput': throughput,
        'peak_memory_mb': np.max(memory_usage),
    }

# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    # Print system info
    print_system_info()
    
    # Path to your exported model
    # Use the portable one if you are testing the same one used in Android
    PTE_MODEL_PATH = "mobilenetv3_qat_acu76.pte" 
    
    if not os.path.exists(PTE_MODEL_PATH):
        print(f"Error: Model file {PTE_MODEL_PATH} not found.")
        exit(1)

    # Initialize Wrapper
    model_wrapper = ExecuTorchWrapper(PTE_MODEL_PATH)
    
    # Get class names (assuming you have 'full_dataset' or 'class_names' defined from your previous code)
    # If not defined in this script, define them manually:
    class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast', 'Leaf scald', 'Sheath Blight']


    # Run evaluation
    # Note: Ensure val_loader_batch_size_1 is defined (batch size MUST be 1 for most exported models)
    results = evaluate_executorch_model(
        model_wrapper=model_wrapper,
        val_loader=val_loader_batch_size_1, # Ensure this loader exists from your data prep code
        class_names=class_names,
        warmup_steps=10
    )
    
    print("\nEvaluation complete! ✅")