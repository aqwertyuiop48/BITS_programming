from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import onnxruntime as ort
import numpy as np

def main():
    # --- Step 1: Define paths and show baseline size ---
    input_model = "models/resnet18.onnx"
    output_model = "models/resnet18_int8.onnx"
    
    if not os.path.exists(input_model):
        print(f"Error: Baseline model {input_model} not found.")
        print("Please run `python scripts/setup_baseline.py` first.")
        return

    size_fp32 = os.path.getsize(input_model) / (1024 * 1024)
    print(f"Baseline FP32 ONNX size: {size_fp32:.2f} MB")

    # --- Step 2: Apply dynamic quantization ---
    print("\nApplying dynamic quantization...")
    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QUInt8  # Quantize weights to 8-bit unsigned integers
    )
    print(f"Quantized model saved to: {output_model}")

    # --- Step 3: Compare file sizes ---
    size_int8 = os.path.getsize(output_model) / (1024 * 1024)
    print(f"\n--- Size Comparison ---")
    print(f"FP32 ONNX size: {size_fp32:.2f} MB")
    print(f"INT8 ONNX size: {size_int8:.2f} MB")
    print(f"Reduction:      {(1 - size_int8 / size_fp32) * 100:.1f}%")

    # --- Step 4: Quick sanity check ---
    print("\n--- Sanity Checking Quantized Model ---")
    try:
        session = ort.InferenceSession(
            output_model,
            providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run([output_name], {input_name: dummy_input})
        print("INT8 model loaded and executed successfully.")
        print("INT8 logits shape:", outputs[0].shape)
    except Exception as e:
        print(f"Error during sanity check: {e}")

if __name__ == "__main__":
    main()
