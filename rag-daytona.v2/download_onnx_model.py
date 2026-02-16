
import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
save_directory = "/home/prometheus/leibniz_agent/TARA-MICROSERVICE/rag-daytona.v2/models/Xenova/paraphrase-multilingual-MiniLM-L12-v2"

print(f"Downloading model {model_id} to {save_directory}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=False)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("✅ Model downloaded and saved successfully!")
