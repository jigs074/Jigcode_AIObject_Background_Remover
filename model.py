import tensorflow_hub as hub

# Correct URL for DeepLabV3 model
model_url = "https://tfhub.dev/tensorflow/deeplab/3"  # Version 2 of DeepLabV3
model = hub.load(model_url)

print("Model loaded successfully!")
