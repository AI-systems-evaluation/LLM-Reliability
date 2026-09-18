"""
Experiment configuration for LLM Reliability project.

Contains definitions for supported LLMs, model descriptors,
and expert evaluation setups (leaderboards).
"""

from typing import List, Tuple, Dict, Any, Union
import numpy as np

# List of supported LLMs (HuggingFace/OpenAI paths)
LLMs: List[str] = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "google/gemma-1.1-7b-it",
    "google/gemma-2-2b-it",
    "google/gemma-7b-it",
    "google/gemma-1.1-2b-it",
    "google/gemma-2b-it",
    'openai/gpt-4.1-2025-04-14',
    'openai/gpt-4.1-mini-2025-04-14',
    'openai/gpt-4.1-nano-2025-04-14',
    'openai/gpt-4o-2024-08-06',
    'openai/gpt-4o-mini-2024-07-18',
    'openai/gpt-5.1-2025-11-13',
    'openai/gpt-5-2025-08-07',
    'openai/gpt-5-mini-2025-08-07',
    'openai/gpt-5-nano-2025-08-07',
    'openai/o3-2025-04-16',
    'openai/o3-mini-2025-01-31',
    'openai/o4-mini-2025-04-16',
]

# Experiment hyperparameters
sample_size: int = 2000
resample_count: int = 16

# Descriptors used for geometric/statistical analysis of embeddings
DESCRIPTORS: List[str] = [
    "Vol", "Area", "Peri", "GKDE", "Dmean", "Dmax",
    "cov_lambda_1", "cov_lambda_2", "cov_lambda_3",
    "cor_lambda_1", "cor_lambda_2", "cor_lambda_3",
]

# Leaderboard data mapping model name to various external expert evaluation scores
LLM_LEADER_BOARD_TABLE: Dict[str, Union[List[str], List[List[Any]]]] = {
    "header": [
        "model",                                   "Average", "IFEval", "BBH",    "MATH",  "GPQA",  "MUSR",  "MMLU-PRO", "HHEM-2.3"
    ],
    "body": [
        ["mistralai/Mixtral-8x7B-Instruct-v0.1",   23.82,     55.99,    29.74,    9.14,    7.05,    11.07,   29.91,      20.1],
        ["mistralai/Mistral-7B-Instruct-v0.1",     12.77,     44.87,    7.65,     2.27,    0,       6.13,    15.72,      np.nan],
        ["mistralai/Mistral-7B-Instruct-v0.2",     18.51,     54.96,    22.91,    3.02,    3.47,    7.61,    19.08,      np.nan],
        ["mistralai/Mistral-7B-Instruct-v0.3",     19.23,     54.65,    25.57,    3.85,    3.91,    4.3,     23.06,      9.5],

        ["meta-llama/Llama-3.1-8B-Instruct",       23.76,     49.22,    29.38,    15.56,   8.72,    8.61,    31.09,      5.4],
        ["meta-llama/Llama-3.2-3B-Instruct",       24.2,      73.93,    24.06,    17.67,   3.8,     1.37,    24.39,      7.9],
        ["meta-llama/Llama-3.2-1B-Instruct",       14.44,     56.98,    8.74,     7.02,    3.36,    2.97,    7.58,       20.7],
        ["meta-llama/Llama-3.3-70B-Instruct",      44.85,     89.98,    56.56,    48.34,   10.51,   15.57,   48.13,      4.0],

        ["google/gemma-1.1-7b-it",                 17.69,     50.39,    15.93,    4.91,    5.82,    11.51,   17.6,       17.0],
        ["google/gemma-2-2b-it",                   17.05,     56.68,    17.98,    0.08,    3.24,    7.08,    17.22,      7.0],
        ["google/gemma-7b-it",                     13.07,     38.68,    11.94,    2.95,    4.59,    12.53,   7.72,       14.8],
        ["google/gemma-1.1-2b-it",                 8.05,      30.67,    5.86,     1.81,    2.57,    2.02,    5.37,       27.8],
        ["google/gemma-2b-it",                     7.49,      26.9,     5.21,     2.04,    3.8,     3.03,    3.92,       np.nan],

        ['openai/gpt-4.1-2025-04-14',              np.nan,    np.nan,   np.nan,   39.58,   65.40,   np.nan,  80.49,      2.0],
        ['openai/gpt-4.1-mini-2025-04-14',         np.nan,    np.nan,   np.nan,   49.38,   67.93,   np.nan,  77.22,      2.2],
        ['openai/gpt-4.1-nano-2025-04-14',         np.nan,    np.nan,   np.nan,   26.46,   50.76,   np.nan,  63.48,      2.0],

        ['openai/gpt-4o-2024-08-06',               np.nan,    np.nan,   np.nan,   11.88,   53.79,   np.nan,  72.56,      1.5],
        ['openai/gpt-4o-mini-2024-07-18',          np.nan,    np.nan,   np.nan,   11.46,   44.19,   np.nan,  62.74,      1.7],

        ['openai/gpt-5.1-2025-11-13',              np.nan,    np.nan,   np.nan,   93.33,   86.62,   np.nan,  86.36,      np.nan],
        ['openai/gpt-5-2025-08-07',                np.nan,    np.nan,   np.nan,   93.37,   85.61,   np.nan,  86.54,      4.9],
        ['openai/gpt-5-mini-2025-08-07',           np.nan,    np.nan,   np.nan,   91.46,   80.30,   np.nan,  82.23,      3.2],
        ['openai/gpt-5-nano-2025-08-07',           np.nan,    np.nan,   np.nan,   81.18,   63.38,   np.nan,  76.07,      4.7],

        ['openai/o3-2025-04-16',                   np.nan,    np.nan,   np.nan,   85.28,   84.09,   np.nan,  85.59,      6.8],
        ['openai/o3-mini-2025-01-31',              np.nan,    np.nan,   np.nan,   86.46,   75.50,   np.nan,  78.69,      np.nan],
        ['openai/o4-mini-2025-04-16',              np.nan,    np.nan,   np.nan,   83.67,   74.50,   np.nan,  80.56,      4.6],
    ]
}

# Internal indexing metrics
experts_I: List[str] = ['SDI', 'CHSM', 'GRS']

# External performance metrics
experts_E: List[str] = ['MMLU-PRO', 'GPQA', 'MATH']