"""
Script to run the Q-Model inference for LLM Reliability.
Combines internal (I) metrics and external (E) leaderboard scores 
using a Bayesian model built with PyMC.
"""

import argparse
import os
import pickle
from typing import Dict, Any, Tuple

import pandas as pd
from scipy import stats
import pymc as pm
import arviz as az

from experiment_config import (
    LLMs,
    LLM_LEADER_BOARD_TABLE, 
    experts_I, 
    experts_E
)


def load_and_preprocess_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, list, int, int]:
    """
    Load data from CSVs and construct a normalized dataframe for the PyMC model.
    """
    K = args.k
    prefix = args.prefix
    
    # Leaderboard table extraction
    llm_arena_score = pd.DataFrame(
        columns=LLM_LEADER_BOARD_TABLE["header"],
        data=LLM_LEADER_BOARD_TABLE["body"]
    ).set_index('model')[experts_E] 
    
    SDI_df = pd.read_csv(os.path.join(prefix, "SDI.csv"))
    CHSM_df = pd.read_csv(os.path.join(prefix, "CHSM.csv"))
    GRS_df = pd.read_csv(os.path.join(prefix, "GRS.csv"))
    
    total_df = (SDI_df.set_index(["K", "llm", "resample"])
                .join(CHSM_df.set_index(["K", "llm", "resample"]))
                .join(GRS_df.set_index(["K", "llm", "resample"]))
                .reset_index()
                .query("K == @K")
                .drop(columns=["K"]))
                
    # Reorient variables so higher is better
    total_df_reorientation = total_df.copy()
    total_df_reorientation['SDI'] = -total_df_reorientation['SDI']
    total_df_reorientation['CHSM'] = -total_df_reorientation['CHSM']
    
    anchor_E_name = args.anchor_E_name
    anchor_E = [0] * len(experts_E)
    if anchor_E_name in experts_E:
        anchor_E[experts_E.index(anchor_E_name)] = 1
    else:
        raise ValueError(f"Anchor '{anchor_E_name}' not found in experts_E {experts_E}")
        
    assert sum(anchor_E) == 1
    
    num_items = len(LLMs)
    num_rating = total_df_reorientation['resample'].unique().shape[0]
    
    # --- Simulated Data Construction ---
    data = pd.DataFrame({
        'expert': [f'I{i}' for llm_name in LLMs for i in range(len(experts_I)) for _ in range(num_rating)] \
                + [f'E{i}' for llm_name in LLMs for i in range(len(experts_E))],
    
        'item':   [i for i in range(num_items) for _ in range(len(experts_I))  for _ in range(num_rating)] \
                + [i for i in range(num_items) for _ in range(len(experts_E))],
    
        'rating': [r for llm_name in LLMs for expert_I in experts_I for r in total_df_reorientation.query("llm == @llm_name")[expert_I].values] \
                + [r for llm_name in LLMs for expert_E in experts_E for r in llm_arena_score.loc[[llm_name], expert_E].values.tolist()]
    })
    
    # --- Normalization (Per Expert) ---
    stats_df = data.groupby('expert')['rating'].agg(['mean', 'std']).fillna(1.0)
    data['norm_rating'] = data.apply(
        lambda x: (x['rating'] - stats_df.loc[x['expert'], 'mean']) / stats_df.loc[x['expert'], 'std'], 
        axis=1
    )
    
    return data, anchor_E, num_items, num_rating


def build_and_run_model(data: pd.DataFrame, anchor_E: list, num_items: int) -> az.InferenceData:
    """
    Construct the PyMC Multi-Item Bayesian Model and execute sampling.
    """
    expert_mapping = {name: i for i, name in enumerate([f'I{k}' for k in range(len(experts_I))] + [f'E{k}' for k in range(len(experts_E))])}
    item_mapping = {i: i for i in range(num_items)}
    
    expert_ids = data['expert'].map(expert_mapping).values
    item_ids = data['item'].map(item_mapping).values
    norm_ratings = data['norm_rating'].values
    
    n_experts_I = len(experts_I)
    n_experts_E = len(experts_E)
    n_items = num_items
    
    with pm.Model() as model:
        # Hyperpriors
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        mu_b = pm.Normal("mu_b", mu=1, sigma=1)
        tau_a = pm.HalfNormal("tau_a", sigma=1)
        tau_b = pm.HalfNormal("tau_b", sigma=1)
    
        # Latent True Scores
        theta = pm.Normal("theta", mu=0, sigma=1, shape=n_items)
    
        # Expert-specific parameters
        a_i = pm.Normal("a_i", mu=mu_a, sigma=tau_a, shape=n_experts_I)
        b_i = pm.HalfNormal("b_i", sigma=tau_b, shape=n_experts_I)
    
        a_Es = []
        b_Es = []
    
        for Ei in range(n_experts_E):
            if anchor_E[Ei]:
                a_Ei = pm.Normal(f"a_e{Ei}", mu=0, sigma=1e-6, shape=1)
                b_Ei = pm.Normal(f"b_e{Ei}", mu=1, sigma=1e-6, shape=1)
            else:
                a_Ei = pm.Normal(f"a_e{Ei}", mu=mu_a, sigma=tau_a, shape=1)
                b_Ei = pm.HalfNormal(f"b_e{Ei}", sigma=tau_b, shape=1)
    
            a_Es.append(a_Ei)
            b_Es.append(b_Ei)
    
        sigma = pm.HalfNormal("sigma", sigma=1, shape=n_experts_I+n_experts_E)
    
        a = pm.math.concatenate([a_i] + a_Es)
        b = pm.math.concatenate([b_i] + b_Es)
    
        mu_x = a[expert_ids] + b[expert_ids] * theta[item_ids]
    
        # Likelihood
        obs = pm.Normal("obs", mu=mu_x, sigma=sigma[expert_ids], observed=norm_ratings)
    
        # Inference
        trace = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True)
        
    return trace


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for running the Q-Model.
    """
    Qmodel_ofile = os.path.join(args.prefix, f"Qmodel-{args.anchor_E_name}.pkl")
    if os.path.exists(Qmodel_ofile):
        print(f"File {Qmodel_ofile} already exists. Skipping.")
        return
    
    print("Preparing data...")
    data, anchor_E, num_items, num_rating = load_and_preprocess_data(args)
    
    print("Building and running the PyMC model...")
    trace = build_and_run_model(data, anchor_E, num_items)
    
    print("Extracting results...")
    posterior_theta = az.summary(trace, var_names=["theta"])
    print("Latent True Scores (Consensus):")
    print(posterior_theta[['mean', 'hdi_3%', 'hdi_97%']])
    
    with open(Qmodel_ofile, "wb") as f:
        pickle.dump({
            "data": data,
            "trace": trace,
            "posterior_theta": posterior_theta
        }, f)
        
    print(f"Results saved to {Qmodel_ofile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-Model inference for LLM Reliability")
    parser.add_argument("--anchor_E_name", type=str, required=True, help="Name of the anchor expert")
    parser.add_argument("--k", type=int, default=30, help="Value of K to filter the dataframe")
    parser.add_argument("--prefix", type=str, default="outputs-rel_indices", help="Prefix directory for outputs and inputs")
    
    args = parser.parse_args()
    main(args)
