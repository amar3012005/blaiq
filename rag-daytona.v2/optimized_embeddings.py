import os
import logging
import torch
import numpy as np
from typing import List, Union
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

logger = logging.getLogger(__name__)

class OptimizedEmbeddings:
    """
    High-performance embedding wrapper for BAAI/bge-m3 using ONNX Runtime.
    Optimized for vCPU environments with sub-second cold starts and low latency.
    """
    
    def __init__(self, model_path: str, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        
        logger.info(f"🚀 Loading optimized ONNX model from {model_path}...")
        try:
            # Suppress regex warning by using clean tokenizer config
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                use_fast=True,
                clean_up_tokenization_spaces=True
            )
            # Load standard ONNX model
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                model_path, 
                provider="CPUExecutionProvider"
            )
            
            # Detect dimension
            test_sentence = "This is a test to detect dimension."
            self.dimension = len(self.embed_query(test_sentence))
            logger.info(f"✅ Optimized ONNX model loaded successfully ({self.dimension}-D)")
        except Exception as e:
            logger.error(f"❌ Failed to load optimized model: {e}")
            raise

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences: Union[str, List[str]], convert_to_numpy: bool = True, normalize_embeddings: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Replacement for SentenceTransformer.encode"""
        if isinstance(sentences, str):
            sentences = [sentences]

        # Tokenize
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        # Inference
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Mean Pooling
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize
        if normalize_embeddings:
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
        if convert_to_numpy:
            return sentence_embeddings.cpu().numpy()
        
        return sentence_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compatible with LangChain style."""
        embeddings = self.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compatible with LangChain style."""
        embedding = self.encode([text])[0]
        return embedding.tolist()
