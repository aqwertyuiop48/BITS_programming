import torch
from torchvision import models
import os
import warnings

def setup():
    print("Setting up baseline model...")
    os.makedirs("models", exist_ok=True)
    
    # Load and save PyTorch model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    pt_path = "models/resnet18_baseline.pt"
    torch.save(model.state_dict(), pt_path)
    pt_size_mb = os.path.getsize(pt_path) / (1024 * 1024)
    print(f"PyTorch model saved: {pt_size_mb:.2f} MB")
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "models/resnet18.onnx"
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX model saved: {onnx_size_mb:.2f} MB")
    print("Baseline PyTorch and ONNX models created in models/ folder.")

if __name__ == "__main__":
    setup()
