import torch 
import torchvision 
import tensorflow as tf 

print("Smart Agri AI Enviroment Check")

print("===================================")
print(f"Pytorch version : {torch.__version__}")
print(f"Torchvision version : {torchvision.__version__}")
print(f"Tensorflow version : {tf.__version__}")
print("===================================")

print(f"CUDA available : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: Not available - using CPU instead")

print("===================================")
print("Enviroment setup successfull!")
