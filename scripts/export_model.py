import torch
import torch.onnx
from torchvision import models
import os
import json
from datetime import datetime

def export_to_onnx(model_path, export_path="exports/pneumonia_model.onnx", input_size=(1, 3, 224, 224)):
    """
    Export PyTorch model to ONNX format for deployment
    
    Args:
        model_path: Path to the trained PyTorch model
        export_path: Path where ONNX model will be saved
        input_size: Input tensor size (batch, channels, height, width)
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Create export directory
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ Model exported to ONNX: {export_path}")
        return export_path
    except Exception as e:
        print(f"❌ Error exporting to ONNX: {e}")
        return None

def export_to_torchscript(model_path, export_path="exports/pneumonia_model_scripted.pt"):
    """
    Export PyTorch model to TorchScript for production deployment
    
    Args:
        model_path: Path to the trained PyTorch model
        export_path: Path where TorchScript model will be saved
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create export directory
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    # Export to TorchScript
    try:
        # Use tracing method
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        scripted_model = torch.jit.trace(model, dummy_input)
        scripted_model.save(export_path)
        
        print(f"✅ Model exported to TorchScript: {export_path}")
        return export_path
    except Exception as e:
        print(f"❌ Error exporting to TorchScript: {e}")
        return None

def create_model_metadata(model_path, accuracy, classes=["NORMAL", "PNEUMONIA"]):
    """
    Create metadata file for the exported model
    
    Args:
        model_path: Path to the model file
        accuracy: Model accuracy on validation set
        classes: List of class names
    """
    metadata = {
        "model_info": {
            "architecture": "MobileNetV2",
            "task": "Pneumonia Detection",
            "classes": classes,
            "num_classes": len(classes),
            "input_size": [224, 224],
            "input_channels": 3
        },
        "performance": {
            "validation_accuracy": float(accuracy),
            "evaluation_date": datetime.now().isoformat()
        },
        "preprocessing": {
            "resize": [224, 224],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        },
        "deployment": {
            "framework": "PyTorch",
            "device_requirements": "CPU/GPU compatible",
            "memory_requirements": "~14MB model size"
        },
        "usage_example": {
            "python_code": """
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load('pneumonia_model.pth'))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict
image = Image.open('xray.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.softmax(output, dim=1)
    
result = ['NORMAL', 'PNEUMONIA'][prediction.argmax().item()]
confidence = prediction.max().item()
"""
        }
    }
    
    # Save metadata
    metadata_path = model_path.replace('.pth', '_metadata.json').replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: {metadata_path}")
    return metadata_path

def export_model_package(model_path, validation_accuracy, export_dir="exports"):
    """
    Create a complete model package ready for deployment
    
    Args:
        model_path: Path to the trained model
        validation_accuracy: Model's validation accuracy
        export_dir: Directory to save all exports
    
    Returns:
        dict: Paths to all exported files
    """
    os.makedirs(export_dir, exist_ok=True)
    
    print("📦 Creating model package...")
    
    # Export to different formats
    onnx_path = export_to_onnx(model_path, os.path.join(export_dir, "pneumonia_model.onnx"))
    torchscript_path = export_to_torchscript(model_path, os.path.join(export_dir, "pneumonia_model_scripted.pt"))
    
    # Create metadata
    metadata_path = create_model_metadata(
        os.path.join(export_dir, "pneumonia_model"),
        validation_accuracy
    )
    
    # Copy original PyTorch model
    import shutil
    pytorch_export_path = os.path.join(export_dir, "pneumonia_model.pth")
    shutil.copy2(model_path, pytorch_export_path)
    
    # Create deployment instructions
    instructions_path = create_deployment_instructions(export_dir)
    
    package_info = {
        'pytorch_model': pytorch_export_path,
        'onnx_model': onnx_path,
        'torchscript_model': torchscript_path,
        'metadata': metadata_path,
        'instructions': instructions_path
    }
    
    print(f"✅ Model package created in: {export_dir}")
    return package_info

def create_deployment_instructions(export_dir):
    """Create deployment instructions file"""
    instructions = """# Pneumonia Detection Model - Deployment Guide

## Model Files
- `pneumonia_model.pth` - Original PyTorch model (requires PyTorch)
- `pneumonia_model.onnx` - ONNX format (cross-platform, use with ONNX Runtime)
- `pneumonia_model_scripted.pt` - TorchScript (PyTorch deployment, no Python dependencies)
- `pneumonia_model_metadata.json` - Model metadata and specifications

## Quick Start

### Using PyTorch Model:
```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load('pneumonia_model.pth'))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict
image = Image.open('chest_xray.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.softmax(output, dim=1)

classes = ["NORMAL", "PNEUMONIA"]
result = classes[prediction.argmax().item()]
confidence = prediction.max().item()
```

### Using ONNX Model:
```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession('pneumonia_model.onnx')

# Preprocess image (same as above)
# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_tensor.numpy()})
```

## Requirements
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- PIL/Pillow
- For ONNX: onnxruntime

## Important Notes
- Input images must be RGB format
- Images are resized to 224x224 pixels
- Use ImageNet normalization values
- This model is for educational purposes only
- Always consult medical professionals for actual diagnosis

## Model Performance
- Validation Accuracy: See metadata file
- Architecture: MobileNetV2 with transfer learning
- Classes: NORMAL, PNEUMONIA
"""
    
    instructions_path = os.path.join(export_dir, "README.md")
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    print(f"📖 Deployment instructions saved to: {instructions_path}")
    return instructions_path

def benchmark_model(model_path, num_runs=100):
    """
    Benchmark model inference speed
    
    Args:
        model_path: Path to the model
        num_runs: Number of inference runs for benchmarking
    """
    import time
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Warm up
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time
    
    print(f"⚡ Model Benchmark Results ({num_runs} runs):")
    print(f"  Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Frames per second: {fps:.1f} FPS")
    print(f"  Device: {device}")
    
    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'fps': fps,
        'device': str(device)
    }

if __name__ == "__main__":
    # Example usage
    model_path = os.path.join("models", "pneumonia_model.pth")
    
    if os.path.exists(model_path):
        print("🚀 Exporting model package...")
        
        # Create full export package
        package_info = export_model_package(model_path, validation_accuracy=0.85)  # Replace with actual accuracy
        
        # Benchmark the model
        benchmark_results = benchmark_model(model_path)
        
        print(f"\n📊 Export Summary:")
        for name, path in package_info.items():
            if path and os.path.exists(path):
                print(f"  ✅ {name}: {path}")
            else:
                print(f"  ❌ {name}: Failed to create")
                
    else:
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first: python src/train.py")