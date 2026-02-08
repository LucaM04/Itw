import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Erkannte GPU: {torch.cuda.get_device_name(0)}")
    print("🚀 Deine RTX 4070 ist bereit zum Lernen!")
else:
    print("⚠️ Läuft nur auf CPU.")
