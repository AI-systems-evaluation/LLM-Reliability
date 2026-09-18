"""
Script to compute or aggregate reliability indices for Algorithm 1.
"""

import argparse
import os
import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import reliability_indexing as ri
from experiment_config import DESCRIPTORS, LLMs


def compute_indices(args: argparse.Namespace) -> None:
    """
    Compute reliability indices (SDI, CHSM, GRS) for a specific value of K.
    Reads pre-computed clustering and shape descriptors.
    """
    K = args.k
    print(f"Computing reliability indices for K={K}")
    os.makedirs(args.outdir, exist_ok=True)

    total_results = []
    
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    
    for resample_seed in tqdm(seeds, desc=f"Loading K={K} Data"):
        file_path = f"{args.prefix}/resample2k3k/kmeans/{K}/alg1-joint-resample2000-seed{resample_seed}.pkl"
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
            
        with open(file_path, "rb") as f:
            total_results += pickle.load(f)

    for result in tqdm(total_results, desc="Assigning Configurations"):
        result.descriptors = DESCRIPTORS

    ri.normalize_global(total_results)

    for result in tqdm(total_results, desc="Computing SDI, CHSM, GRS"):
        ri.compute_SDI(result)
        ri.compute_CHSM(result)
        ri.compute_GRS(result)

        # Cleanup large attributes to save memory
        del result.viz0
        del result.llm_vizs

    rel_indices: List[Dict[str, Any]] = [
        {
            "K": K,
            "resample": i,
            "SDI": res.SDI,
            "CHSM_input": res.CHSM_input,
            "CHSM_output": res.CHSM_output,
            "GRS_input": res.GRS_input,
            "GRS_output": res.GRS_output,
            "index_dataframe": res.index_dataframe,
        }
        for i, res in enumerate(total_results, start=1)
    ]

    out_path = os.path.join(args.outdir, f"rel_indices_K={K}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(rel_indices, f)
    print(f"Saved {out_path}")


def aggregate_indices(args: argparse.Namespace) -> None:
    """
    Aggregate computed reliability metrics across multiple K values 
    into CSV files (SDI.csv, CHSM.csv, GRS.csv).
    """
    print("Aggregating reliability indices...")
    os.makedirs(args.outdir, exist_ok=True)
    
    rel_indices = []
    for K in args.k_values:
        in_path = os.path.join(args.outdir, f"rel_indices_K={K}.pkl")
        if not os.path.exists(in_path):
            print(f"Warning: {in_path} not found. Skipping.")
            continue
        with open(in_path, "rb") as f:
            rel_indices += pickle.load(f)

    if not rel_indices:
        print("No data to aggregate.")
        return

    SDI_df = pd.concat([
        pd.DataFrame(row["SDI"].mean(axis=1).round(9), columns=['SDI'])
        .assign(llm=LLMs, resample=row["resample"], K=row["K"])
        for row in rel_indices
    ])

    CHSM_df = pd.concat([
        pd.DataFrame(np.array([row["CHSM_input"].mean(), *row["CHSM_output"].mean(axis=(0, 2, 3)).tolist()]).round(9), columns=['CHSM'])
        .assign(llm=['input'] + LLMs, resample=row["resample"], K=row["K"])
        for row in rel_indices
    ])

    GRS_df = pd.concat([
        pd.DataFrame(np.array([row["GRS_input"]] + row["GRS_output"].tolist()).round(9), columns=['GRS'])
        .assign(llm=['input'] + LLMs, resample=row["resample"], K=row["K"])
        for row in rel_indices
    ])

    SDI_df.to_csv(os.path.join(args.outdir, "SDI.csv"), index=False)
    CHSM_df.to_csv(os.path.join(args.outdir, "CHSM.csv"), index=False)
    GRS_df.to_csv(os.path.join(args.outdir, "GRS.csv"), index=False)
    print(f"Saved SDI.csv, CHSM.csv, and GRS.csv to {args.outdir}")


def main() -> None:
    """
    Main entry point for Algorithm 1 indices computation and aggregation.
    """
    parser = argparse.ArgumentParser(description="Compute or aggregate reliability indices for Algorithm 1.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Compute action
    parser_compute = subparsers.add_parser("compute", help="Compute reliability indices for a specific K")
    parser_compute.add_argument("--k", type=int, required=True, help="Value for K (e.g., 10, 20, 30, 40)")
    parser_compute.add_argument("--prefix", type=str, default=".", help="Prefix path for data")
    parser_compute.add_argument("--outdir", type=str, default="outputs-rel_indices", help="Output directory")
    parser_compute.add_argument("--seed_start", type=int, default=1001, help="Starting random seed")
    parser_compute.add_argument("--seed_count", type=int, default=16, help="Number of seeds to process")

    # Aggregate action
    parser_aggregate = subparsers.add_parser("aggregate", help="Aggregate metrics across K values to CSV")
    parser_aggregate.add_argument("--k_values", type=int, nargs="+", default=[10, 20, 30, 40], help="List of K values to aggregate")
    parser_aggregate.add_argument("--outdir", type=str, default="outputs-rel_indices", help="Output directory")

    args = parser.parse_args()

    if args.action == "compute":
        compute_indices(args)
    elif args.action == "aggregate":
        aggregate_indices(args)


if __name__ == "__main__":
    main()
