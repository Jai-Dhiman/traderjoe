#!/usr/bin/env python3
"""
EmbeddingGemma service for TraderJoe
Provides embedding generation using sentence-transformers library
"""

import sys
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Singleton service for generating embeddings with EmbeddingGemma"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the embedding model (lazy loading)"""
        if self._model is None:
            self.load_model()

    def load_model(self):
        """Load the EmbeddingGemma model from HuggingFace"""
        try:
            print(json.dumps({"status": "loading", "message": "Loading EmbeddingGemma model"}), file=sys.stderr, flush=True)
            self._model = SentenceTransformer("google/embeddinggemma-300m")
            print(json.dumps({"status": "loaded", "message": "Model loaded successfully"}), file=sys.stderr, flush=True)
        except Exception as e:
            print(json.dumps({"status": "error", "message": f"Failed to load model: {str(e)}"}), file=sys.stderr, flush=True)
            raise

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Generate embedding
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Generate embeddings
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def dimension(self) -> int:
        """Get the embedding dimension"""
        return 768


def main():
    """Main entry point for command-line usage"""
    service = EmbeddingService()

    # Read commands from stdin
    for line in sys.stdin:
        try:
            command = json.loads(line.strip())
            cmd_type = command.get("type")

            if cmd_type == "embed":
                text = command.get("text")
                if not text:
                    raise ValueError("Missing 'text' field")

                embedding = service.embed(text)
                response = {"status": "success", "embedding": embedding}
                print(json.dumps(response), flush=True)

            elif cmd_type == "embed_batch":
                texts = command.get("texts")
                if not texts or not isinstance(texts, list):
                    raise ValueError("Missing or invalid 'texts' field")

                embeddings = service.embed_batch(texts)
                response = {"status": "success", "embeddings": embeddings}
                print(json.dumps(response), flush=True)

            elif cmd_type == "dimension":
                dim = service.dimension()
                response = {"status": "success", "dimension": dim}
                print(json.dumps(response), flush=True)

            elif cmd_type == "exit":
                break

            else:
                response = {"status": "error", "message": f"Unknown command type: {cmd_type}"}
                print(json.dumps(response), flush=True)

        except Exception as e:
            response = {"status": "error", "message": str(e)}
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
