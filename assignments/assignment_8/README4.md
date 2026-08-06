# M8 Compression Lab: Quantization & Benchmarking

This repository contains the lab materials for Module 8, covering model compression through dynamic quantization and benchmarking the results to make informed deployment decisions.

## Repository Structure

```
Lab_8/
├── data/
│   └── val_samples/
│       ├── labels.json         # Labels for validation images
│       └── val_image_0.jpg     # Placeholder for validation image
├── models/                     # Output directory for saved models
├── scripts/
│   ├── setup_baseline.py       # Creates baseline PyTorch and ONNX models
│   ├── m8_l1_quantize_onnx.py  # Lab 1: Applies dynamic quantization
│   └── m8_l2_benchmark.py      # Lab 2: Benchmarks FP32 vs INT8 models
├── m8_lab_details.md           # Trainer's script and lab guide
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Baseline Models

```bash
python scripts/setup_baseline.py
```
This creates the baseline `resnet18.onnx` file in the `models/` directory.

### 3. Lab 1: Apply Quantization

```bash
python scripts/m8_l1_quantize_onnx.py
```
This applies dynamic quantization to create `resnet18_int8.onnx`.

### 4. Lab 2: Run Benchmarks

```bash
python scripts/m8_l2_benchmark.py
```
This compares the FP32 and INT8 models on size, latency, and accuracy.

