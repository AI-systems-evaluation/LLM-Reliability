"""
Script for generating and saving text embeddings using SentenceTransformer.
"""

import argparse
import json
import logging
import os
from typing import Optional, List, Tuple, Any

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)


class Embedder:
    """
    A class to generate and manage text embeddings using a specified model.
    """
    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        embed_device: str = "cpu",
        embed_batch_size: int = 64,
        embed_max_seq_length: int = 512,
        embed_agg_strategy: Optional[Any] = None,
    ):
        """
        Initialize the Embedder with the specified configuration.
        """
        self.embed_model_name = embed_model_name
        self.embed_device = embed_device
        self.embed_batch_size = embed_batch_size
        self.embed_max_seq_length = embed_max_seq_length
        self.embed_agg_strategy = embed_agg_strategy

        self.embeddings: Optional[np.ndarray] = None
        self.texts: Optional[List[str]] = None

        self.embed_model = SentenceTransformer(
            self.embed_model_name, device=self.embed_device
        )
        self.embed_model.max_seq_length = self.embed_max_seq_length

    def fit(self, texts: List[str], embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Assign texts and compute embeddings if they are not provided.
        """
        self.texts = texts

        if embeddings is None:
            logging.info("Embedding texts...")
            self.embeddings = self.embed(texts)
        else:
            logging.info("Using precomputed embeddings...")
            self.embeddings = embeddings
        
        return self.embeddings

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for the provided list of texts.
        """
        embeddings = self.embed_model.encode(
            texts,
            batch_size=self.embed_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings

    def save(self, folder: str) -> None:
        """
        Save the computed embeddings and the corresponding texts to the specified folder.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(f"{folder}/embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)

        with open(f"{folder}/texts.json", "w") as f:
            json.dump(self.texts, f)

    def load(self, folder: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load embeddings and texts from the specified folder.
        """
        self.embeddings = np.load(f"{folder}/embeddings.npy")

        with open(f"{folder}/texts.json", "r") as f:
            self.texts = json.load(f)
            
        return self.embeddings, self.texts


def main():
    parser = argparse.ArgumentParser(description="Generate text embeddings.")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name or path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompt")
    parser.add_argument("--response_column", type=str, default="response", help="Column name for response")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder to save embeddings")
    parser.add_argument("--embed_model_name", type=str, default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--embed_device", type=str, default="cpu", help="Device to run embedding model")
    parser.add_argument("--embed_batch_size", type=int, default=64, help="Batch size for embedding")
    parser.add_argument("--embed_max_seq_length", type=int, default=512, help="Max sequence length")

    args = parser.parse_args()

    logging.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    
    logging.info(f"Extracting columns: {args.prompt_column} and {args.response_column}")
    texts = []
    for item in dataset:
        prompt = str(item.get(args.prompt_column, ""))
        response = str(item.get(args.response_column, ""))
        text = f"{prompt}\n{response}".strip()
        texts.append(text)

    logging.info("Initializing embedding model...")
    embedder = Embedder(
        embed_model_name=args.embed_model_name,
        embed_device=args.embed_device,
        embed_batch_size=args.embed_batch_size,
        embed_max_seq_length=args.embed_max_seq_length,
    )

    logging.info("Generating embeddings...")
    embedder.fit(texts)

    logging.info(f"Saving embeddings and texts to {args.output_folder}")
    embedder.save(args.output_folder)
    logging.info("Done!")


if __name__ == "__main__":
    main()
