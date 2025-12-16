import argparse
import sys
import os
import torch
from PIL import Image
from executorch.runtime import Runtime
from helper import val_transforms
from rice_dataset import RiceGrainDataset

def main():
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Rice Grain Classification Prediction")
    
    # Positional argument: Image path
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    
    # Optional arguments for flexibility
    parser.add_argument("--model", type=str, default="models/rice_model_ptq_optimized_99.pte", help="Path to .pte model file")
    parser.add_argument("--dataset_root", type=str, default="./rice-image-dataset/Rice_Image_Dataset", help="Path to dataset root")

    args = parser.parse_args()

    # 1. Validate Image Path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at '{args.image_path}'")
        sys.exit(1)

    # 2. Setup Class Mappings
    # We try to load from dataset class, but fallback to hardcoded if dataset folder is missing
    try:
        dataset = RiceGrainDataset(root_dir=args.dataset_root, transform=None)
        idx_to_class = dataset.idx_to_class
    except Exception:
        # Fallback classes based on your README
        idx_to_class = {0: 'Arborio', 1: 'Basmati', 2: 'Ipsala', 3: 'Jasmine', 4: 'Karacadag'}

    # 3. Load and Transform Image
    try:
        image = Image.open(args.image_path).convert("RGB")
        input_tensor = val_transforms(image)
        # Add batch dimension → [1, C, H, W]
        input_tensor = input_tensor.unsqueeze(0)
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

    # 4. Run Inference
    try:
        runtime = Runtime.get()
        program = runtime.load_program(args.model)
        method = program.load_method("forward")

        with torch.no_grad():
            outputs = method.execute([input_tensor])
            logits = outputs[0]
            predicted_idx = torch.argmax(logits, dim=1).item()

        predicted_class = idx_to_class[predicted_idx]

        print("Prediction Results")
        print("------------------")
        print(f"Input Image        : {args.image_path}")
        print(f"Predicted Class ID : {predicted_idx}")
        print(f"Predicted Class    : {predicted_class}")

    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()