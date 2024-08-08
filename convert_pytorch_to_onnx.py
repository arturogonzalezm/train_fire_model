# import torch
# from torchvision import models
#
# # Load your PyTorch model
# model_path = 'fire_detection_model.pth'
# model = models.resnet50(weights=None)
# num_features = model.fc.in_features
# model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: fire and non-fire
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
# model.eval()
#
# # Dummy input for the model
# dummy_input = torch.randn(1, 3, 224, 224)
#
# # Export the model to ONNX format
# onnx_model_path = 'fire_detection_model.onnx'
# torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)
# print(f"Model successfully converted to {onnx_model_path}")



import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Step 1: Load your PyTorch model
model = torch.load('fire_detection_model.pth')
model.eval()

# Step 2: Export the model to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'fire_detection_model.onnx', input_names=['input'], output_names=['output'])

# Step 3: Load the ONNX model
onnx_model = onnx.load('fire_detection_model.onnx')

# Step 4: Convert ONNX to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph('saved_model')

# Step 5: Convert TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()

# Step 6: Save the TensorFlow Lite model
with open('fire_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)
