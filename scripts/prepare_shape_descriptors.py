"""
Script for preparing shape descriptors for LLM Reliability.
Handles dataset loading, embedding generation, initial sampling, and clustering.
"""

import argparse
import os
import pickle
from functools import partial
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import reliability_indexing as ri
from experiment_config import LLMs
from prepare_embeddings import Embedder


def load_pickle(filepath: str) -> Any:
    """
    Utility function to load a pickle file.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load prompt and response datasets, and merge them into a single DataFrame.
    """
    loaded_data = []
    prompt_dataset = load_dataset(args.prompt_dataset, split=args.split).to_pandas().rename(columns={"id": "prompt_id"})
    
    for llm in LLMs:
        print(f"Loading data from {llm}")
        loaded_data.append(load_dataset(args.response_dataset, data_dir=llm, split=args.split).to_pandas())
    
    # Merge response data with prompts
    loaded_data = pd.merge(pd.concat(loaded_data).reset_index(), prompt_dataset, how="left", on="prompt_id")
    
    return loaded_data


def generate_embeddings_dict(df: pd.DataFrame, embeddings_dir: str) -> Dict[str, np.ndarray]:
    """
    Generate or load embeddings for unique texts in the DataFrame.
    """
    os.makedirs(embeddings_dir, exist_ok=True)
    embedder = Embedder()
    embeddings_file = os.path.join(embeddings_dir, "embeddings.npy")
    
    if not os.path.exists(embeddings_file):
        print("Embeddings not found. Generating new embeddings...")
        unique_texts = list(set(df["prompt"]) | set(df["response"]))
        embedder.fit(unique_texts)
        embedder.save(embeddings_dir)
    else:
        print(f"Loading existing embeddings from {embeddings_dir}...")
        embedder.load(embeddings_dir)

    # Create mapping from text to embedding
    return {text: emb for text, emb in zip(embedder.texts, embedder.embeddings)}


def perform_initial_sampling(df: pd.DataFrame, embeddings_dict: Dict[str, np.ndarray], args: argparse.Namespace) -> None:
    """
    Perform the initial sampling step and save DataStateSample instances to the initial directory.
    """
    os.makedirs(args.initial_dir, exist_ok=True)
    seeds = range(args.seed_start, args.seed_start + args.seed_count)

    for resample_seed in tqdm(seeds, desc="Initial Sampling (Seed)"):
        outfile = os.path.join(args.initial_dir, f"alg1-joint-resample{args.sample_size}-seed{resample_seed}.pkl")
        if os.path.exists(outfile):
            continue

        total_results = [
            ri.DataStateSample(
                df, 
                embeddings_dict,
                random_state=np.random.RandomState(resample_seed),
                sample_size=args.sample_size
            )
        ]
        with open(outfile, "wb") as f:
            pickle.dump(total_results, f)


def process_models(args: argparse.Namespace) -> None:
    """
    Run k-means clustering and compute shape descriptors for the sampled data.
    """
    seeds = range(args.seed_start, args.seed_start + args.seed_count)

    for resample_seed in tqdm(seeds, desc="Processing Models (Seed)"):
        for K in tqdm(args.k_values, desc="K-means (K)", leave=False):
            outdir = os.path.join(args.output_dir, "kmeans", str(K))
            os.makedirs(outdir, exist_ok=True)
            
            ofile = os.path.join(outdir, f"alg1-joint-resample{args.sample_size}-seed{resample_seed}.pkl")
            if os.path.exists(ofile):
                print(f"Skipping {ofile} (already exists)")
                continue

            initial_file = os.path.join(args.initial_dir, f"alg1-joint-resample{args.sample_size}-seed{resample_seed}.pkl")
            total_results = load_pickle(initial_file)

            for result in tqdm(total_results, desc="clustering", leave=False):
                # We rename the monkey-patch method call appropriately if needed
                # Here we pass result as data_state (the first arg) to ri.cluster
                ri.cluster(result, partial(ri.kmean, K=K))
                
            for result in tqdm(total_results, desc="calc_shape_desc", leave=False):
                ri.prepare_shape_descriptor_samples(result)
                
            with open(ofile, "wb") as f:
                pickle.dump(total_results, f)


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for preparing shape descriptors.
    """
    print(f"Loading dataset {args.prompt_dataset} and {args.response_dataset}...")
    df = load_data(args)
    embeddings_dict = generate_embeddings_dict(df, args.embeddings_dir)

    print("Step 1: Initial sampling...")
    perform_initial_sampling(df, embeddings_dict, args)

    print("Step 2: Clustering and calculating shape descriptors...")
    process_models(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare shape descriptors for LLM Reliability")
    parser.add_argument("--prompt_dataset", type=str, default="AI-systems-evaluation/paraphrased_prompt")
    parser.add_argument("--response_dataset", type=str, default="AI-systems-evaluation/paraphrased_prompt_response")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    parser.add_argument("--initial_dir", type=str, default="resample2k3k-initial")
    parser.add_argument("--output_dir", type=str, default="resample2k3k")
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--seed_start", type=int, default=1001)
    parser.add_argument("--seed_count", type=int, default=16)
    parser.add_argument("--k_values", type=int, nargs="+", default=[10, 20, 30, 40])

    args = parser.parse_args()
    main(args)
