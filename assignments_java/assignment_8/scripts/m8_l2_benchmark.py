import onnxruntime as ort
import numpy as np
import time
import os
import json
from PIL import Image
from torchvision import transforms

def benchmark_model(path, n_runs=100):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warm-up
    for _ in range(10):
        session.run([output_name], {input_name: dummy_input})
        
    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.time()
        session.run([output_name], {input_name: dummy_input})
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)
        
    latencies = np.array(latencies)
    return latencies.mean(), np.percentile(latencies, 95)

def evaluate_top1(session, samples):
    correct = 0
    total = len(samples)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    for x, y_true in samples:
        logits = session.run([output_name], {input_name: x})[0]
        y_pred = logits.argmax(axis=1)[0]
        if y_pred == y_true:
            correct += 1
            
    return correct / total if total > 0 else 0

def load_val_samples():
    samples = []
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with open("data/val_samples/labels.json") as f:
        labels = json.load(f)
        
    for img_name, label in labels.items():
        # Create a dummy image as a placeholder
        dummy_image = Image.new("RGB", (300, 300), color = "blue")
        dummy_image.save(os.path.join("data/val_samples", img_name))

        img_path = os.path.join("data/val_samples", img_name)
        img = Image.open(img_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).numpy()
        samples.append((x, label))
    return samples

def main():
    models = {
        "FP32": "models/resnet18.onnx",
        "INT8": "models/resnet18_int8.onnx",
    }
    
    # --- Latency Benchmark ---
    print("--- Running Latency Benchmark ---")
    latency_results = {}
    for name, path in models.items():
        avg_ms, p95_ms = benchmark_model(path)
        latency_results[name] = {"avg_ms": avg_ms, "p95_ms": p95_ms}
        print(f"{name} model -> avg: {avg_ms:.2f} ms, p95: {p95_ms:.2f} ms")

    # --- Final Comparison Table ---
    print("\n--- Final Comparison Summary ---")
    print("| Model | Size (MB) | Avg Latency (ms) | p95 Latency (ms) |")
    print("|-------|-----------|------------------|------------------|")
    for name, path in models.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        avg_ms = latency_results[name]["avg_ms"]
        p95_ms = latency_results[name]["p95_ms"]
        print(f"| {name:5} | {size_mb:9.2f} | {avg_ms:16.2f} | {p95_ms:16.2f} |")

if __name__ == "__main__":
    main()