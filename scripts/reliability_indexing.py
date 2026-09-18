"""
Reliability indexing utilities for LLM Reliability project.
Contains mathematical operations, clustering, shape descriptors computation,
and core index evaluation metrics (SDI, CHSM, GRS).
"""

from collections import defaultdict
from typing import Callable, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from umap import UMAP

import scipy
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans

from tqdm.auto import tqdm
from sklearn.covariance import EmpiricalCovariance
from scipy.linalg import eigvals

import datasets
datasets.logging.disable_progress_bar()


DEFAULT_DESCRIPTORS = [
    "Vol", "Area", "Peri", "GKDE", "Dmean", "Dmax",
    "cov_lambda_1", "cov_lambda_2", "cov_lambda_3",
    "cor_lambda_1", "cor_lambda_2", "cor_lambda_3",
]


def ConvexHullVol_sample(sample: List[np.ndarray], idx: List[int]) -> float:
    return np.mean([ConvexHullVol(X[idx]) for X in sample], axis=0)


def ConvexHullArea_sample(sample: List[np.ndarray], idx: List[int]) -> float:
    return np.mean([ConvexHullArea(X[idx]) for X in sample], axis=0)


def gaussian_kde_vol_sample(sample: List[np.ndarray], idx: List[int]) -> float:
    return np.mean([gaussian_kde_vol(X[idx]) for X in sample], axis=0)


def ConvexHullVol(X: np.ndarray) -> float:
    return ConvexHull(X).volume if len(X) and len(X) > len(X[0]) else 0


def ConvexHullArea(X: np.ndarray) -> float:
    return ConvexHull(X).area if len(X) and len(X) > len(X[0]) else 0


def gaussian_kde_vol(sample00: np.ndarray, k: int = 100, quantile: float = 0.95) -> float:
    """
    Compute the Gaussian KDE volume for a given sample, thresholded by quantile.
    """
    X = sample00[:, 0]
    Y = sample00[:, 1]
    xy = np.vstack([X, Y])
    kde = gaussian_kde(xy, bw_method='scott')

    dX = max(X) - min(X)
    dY = max(Y) - min(Y)
    rd = 1

    x_grid = np.linspace(min(X) - rd * dX, max(X) + rd * dX, k)
    y_grid = np.linspace(min(Y) - rd * dY, max(Y) + rd * dY, k)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    Z = kde(positions)

    dx = (x_grid.max() - x_grid.min()) / (k - 1)
    dy = (y_grid.max() - y_grid.min()) / (k - 1)
    cell_area = dx * dy
    
    Z_flat = Z.ravel()
    sorted_indices = np.argsort(Z_flat)[::-1]
    Z_sorted = Z_flat[sorted_indices]

    cumulative_prob = np.cumsum(Z_sorted * cell_area)
    cumulative_prob /= cumulative_prob[-1]

    try:
        threshold_index = np.where(cumulative_prob >= quantile)[0][0]
        density_threshold = Z_sorted[threshold_index]
    except IndexError:
        return 0
    
    qcut = density_threshold
    ratio = (Z > qcut).mean()
    
    return (2 * rd + 1)**2 * dX * dY * ratio


def find_center_index(points: np.ndarray) -> int:
    """Find the index of the point closest to the mean center."""
    return int(np.argmin(((np.asarray(points) - np.mean(points, axis=0, keepdims=True))**2).sum(axis=1)))


def empirical_covariance_eigenvalues(X: np.ndarray, k: int = 8) -> List[float]:
    """Compute the top k eigenvalues of the empirical covariance matrix."""
    cov = EmpiricalCovariance().fit(X).covariance_
    return sorted((eigvals(cov).real), reverse=True)[:k]


def empirical_correlation_eigenvalues(X: np.ndarray, k: int = 8) -> List[float]:
    """Compute the top k eigenvalues of the empirical correlation matrix."""
    corr = np.corrcoef(X.T)
    return sorted((eigvals(corr).real), reverse=True)[:k]


class JointViz:
    """Stores embeddings, texts, and 2D/3D projections for visualization/analysis."""
    def __init__(self) -> None:
        self.embeddings: np.ndarray = None
        self.texts: List[str] = None
        self.projections_euclidean: List[np.ndarray] = None
        self.projections3D_euclidean: List[np.ndarray] = None

    def select(self, idxs: List[int]) -> None:
        """Subset the current viz by the provided indices."""
        self.embeddings = self.embeddings[idxs]
        self.texts = [self.texts[i] for i in idxs]
               
    def estimate_projections(self, sample_size: int = 8) -> None:
        """Estimate 2D and 3D UMAP projections."""
        self.projections_euclidean = [UMAP(n_components=2, metric="euclidean", random_state=None).fit(self.embeddings).embedding_ for _ in tqdm(range(sample_size), desc="umap")] 
        self.projections3D_euclidean = [UMAP(n_components=3, metric="euclidean", random_state=None).fit(self.embeddings).embedding_ for _ in tqdm(range(sample_size), desc="umap")]


class DataStateSample:
    """
    Maintains the state of sampled data, embeddings, and mathematical metrics.
    """
    def __init__(self, df: pd.DataFrame, embeddings: dict, descriptors: List[str] = DEFAULT_DESCRIPTORS, random_state: np.random.RandomState = None, sample_size: Optional[int] = None):
        self.descriptors = descriptors
        self.random_state = random_state if random_state is not None else np.random.RandomState()
        self.sample_size = sample_size
        
        self.results = dict()
        self.load_viz_data(df, embeddings)
        self.sampling()
        
    def load_viz_data(self, df: pd.DataFrame, embeddings: dict) -> None:
        """Load and structure prompt and response data into visualizations."""
        self.LLMs = df['LLM'].unique().tolist()
        print("processing", df['original_dataset'].unique().tolist())
        
        original_item_map = {}
        for _, row in df.iterrows():
            prompt = str(row['prompt']).strip()
            original_item_map[prompt] = (row['original_context'], row['original_question'])
            
        response_incomplete = {prompt: len(self.LLMs) for prompt in original_item_map}
        prompt_resp_map = {}
        
        for _, row in df.iterrows():
            prompt = str(row['prompt']).strip()
            llm = row['LLM']
            response = str(row['response']).strip() if pd.notnull(row['response']) else ""
            
            if response:
                response_incomplete[prompt] -= 1
            prompt_resp_map[(llm, prompt)] = response
            
        self.viz0 = JointViz()
        self.viz0.texts = [p for p in original_item_map.keys() if not response_incomplete[p]]
        self.viz0.embeddings = np.array([embeddings[p] for p in self.viz0.texts])
        
        emb_size = self.viz0.embeddings.shape[1]
        texts_map = defaultdict(list)
        for idx, vtext in enumerate(self.viz0.texts):
            texts_map[original_item_map[vtext]].append(vtext)
            
        self.viz0.embeddings = list(self.viz0.embeddings)
        viz0_txt2emb = {txt: emb for txt, emb in zip(self.viz0.texts, self.viz0.embeddings)}
        self.max_sampling_rounds = max(len(tx) for tx in texts_map.values()) - 1
        non_makeup_size = len(self.viz0.texts)
        
        for original_text, para_texts in texts_map.items():
            makeup = list(self.random_state.choice(para_texts, size=self.max_sampling_rounds+1-len(para_texts), replace=True))
            self.viz0.texts += makeup
            self.viz0.embeddings += [viz0_txt2emb[txt] + self.random_state.randn(emb_size)*1e-3 for txt in makeup]
            
        self.viz0.embeddings = np.array(self.viz0.embeddings)
        
        points_sets = defaultdict(list)
        for idx, vtext in enumerate(self.viz0.texts):
            points_sets[original_item_map[vtext]].append(idx)
            
        points_sets = list(points_sets.values())
        assert self.max_sampling_rounds == max(len(ps) for ps in points_sets) - 1
        self.random_state.shuffle(points_sets)
        
        if self.sample_size is not None:
            points_sets = points_sets[:self.sample_size]
            
        self.llm_vizs = []
        for name in tqdm(self.LLMs, desc="LLMs"):
            llm_viz = JointViz()
            llm_viz.texts = [prompt_resp_map[(name, text)] for text in self.viz0.texts]
            
            llm_viz.embeddings = np.array([
                embeddings[text] + (idx >= non_makeup_size) * self.random_state.randn(emb_size)*1e-3 
                for idx, text in enumerate(llm_viz.texts)
            ])
            self.llm_vizs.append(llm_viz)
            
        self.index_dataframe = pd.DataFrame(index=range(len(self.viz0.texts)), columns=["root_set"])
        
        self.index_dataframe['root_set'] = -1
        for idx, ps in enumerate(points_sets):
            self.index_dataframe.loc[ps, "root_set"] = idx
            
        self.index_dataframe = self.index_dataframe.query("root_set > -1")
        
        for viz in tqdm(self.llm_vizs + [self.viz0], desc="proj"):
            viz.select(self.index_dataframe.index)
            viz.estimate_projections()
            
        self.index_dataframe = self.index_dataframe.set_index(np.arange(len(self.index_dataframe)))
    
    def sampling(self) -> None:
        """Mark centers and assign sampling rounds for the dataset."""
        self.index_dataframe["is_center"] = 0
        centers = np.array([df.index.values[find_center_index(self.viz0.embeddings[df.index.values])] 
                            for g, df in self.index_dataframe.groupby("root_set")])
        self.index_dataframe.loc[centers, "is_center"] = 1
    
        self.index_dataframe['sampling_round'] = -1
        self.index_dataframe.loc[self.index_dataframe.query('is_center == 1').index, 'sampling_round'] = 0
        
        for g, df in self.index_dataframe.query('is_center == 0').groupby('root_set'):
            indx = list(df.index.values)
            self.random_state.shuffle(indx)
            self.index_dataframe.loc[indx, 'sampling_round'] = np.arange(1, 1 + len(indx))


# --- Standalone DataStateSample Modifiers (Refactored to take `data_state` instead of `self`) ---

def kmean(data_state: DataStateSample, K: int = 30) -> None:
    """Apply KMeans clustering to the center points of the given data state."""
    data_state.index_dataframe["cluster"] = -1
    centers = data_state.index_dataframe.query("is_center == 1").index.values
    
    kmean_model = KMeans(K)
    clusters = kmean_model.fit_predict(data_state.viz0.embeddings[centers])

    for i, cluster_label in enumerate(clusters):
        root_set = data_state.index_dataframe.loc[centers[i], "root_set"]
        data_state.index_dataframe.loc[data_state.index_dataframe.query("root_set == @root_set").index, "cluster"] = cluster_label


def cluster(data_state: DataStateSample, clustering_fn: Callable) -> None:
    """Generic wrapper to apply a clustering function to a data state."""
    clustering_fn(data_state)
    

def normalize_dfs(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Normalize a list of dataframes by dividing by their concatenated standard deviation."""
    scale = pd.concat(dfs).std()
    return [df / scale for df in dfs]


def prepare_shape_descriptor_samples(data_state: DataStateSample) -> None:
    """Compute and store shape descriptors for inputs and outputs over all sampling rounds."""
    data_state.input_descriptor_samplings = []
    data_state.output_descriptor_samplings = []
    
    for sampling_round in tqdm(range(1 + data_state.max_sampling_rounds), desc="resample shape desc"):
        points_df = data_state.index_dataframe.query('sampling_round == @sampling_round')
        
        data_state.input_descriptor_samplings.append(
            compute_shape_descriptor(data_state, points_df, data_state.viz0)
        )
        
        data_state.output_descriptor_samplings.append([
            compute_shape_descriptor(data_state, points_df, llm_viz) 
            for llm_viz in data_state.llm_vizs
        ])

    estimate_empirical_covariance_matrix(data_state)

    
def estimate_empirical_covariance_matrix(data_state: DataStateSample) -> None:
    """Estimate and attach the empirical covariance matrix to the data state."""
    all_shape_descriptor_vectors = []
    
    for df in data_state.input_descriptor_samplings:
        all_shape_descriptor_vectors += list(df[data_state.descriptors].values)
    
    for llm_dfs in data_state.output_descriptor_samplings:
        for df in llm_dfs:
            all_shape_descriptor_vectors += list(df[data_state.descriptors].values)
            
    all_shape_descriptor_vectors = np.array(all_shape_descriptor_vectors)
    
    data_state.empirical_covariance_matrix = EmpiricalCovariance().fit(
        all_shape_descriptor_vectors
    ).covariance_
    
    data_state.empirical_covariance_matrix_addI_inv = scipy.linalg.inv(
        data_state.empirical_covariance_matrix + 1e-3 * np.eye(data_state.empirical_covariance_matrix.shape[0])
    )


def normalize_global(data_states: List[DataStateSample]) -> pd.Series:
    """Globally normalize shape descriptors across multiple DataStateSamples."""
    scale = pd.concat([
        df for data_state in data_states for df in data_state.input_descriptor_samplings
    ] + [
        df for data_state in data_states for dfs in data_state.output_descriptor_samplings for df in dfs
    ]).std()
    
    cov_lambda_scale = scale['cov_lambda_1']
    cor_lambda_scale = scale['cor_lambda_1']
    for col in scale.index:
        if 'cov_lambda_' in col:
            scale[col] = cov_lambda_scale
        if 'cor_lambda_' in col:
            scale[col] = cor_lambda_scale

    for data_state in data_states:
        data_state.input_descriptor_samplings = [df / scale for df in data_state.input_descriptor_samplings]
        data_state.output_descriptor_samplings = [[df / scale for df in dfs] for dfs in data_state.output_descriptor_samplings]
            
        estimate_empirical_covariance_matrix(data_state)

    return scale


def compute_SDI(data_state: DataStateSample) -> None:
    """Compute SDI (Semantic Drift Index) and attach to the data state."""
    wjs = data_state.index_dataframe.query("is_center == 1")["cluster"].value_counts(normalize=True).to_dict()
    data_state.SDI = np.array([
        [np.sqrt(np.sum([
            wjs[cluster_label] * mahalanobis(
                data_state.input_descriptor_samplings[0].loc[cluster_label, data_state.descriptors].values,
                output_0_descriptor_l.loc[cluster_label, data_state.descriptors].values,
                data_state.empirical_covariance_matrix_addI_inv
            )**2 for cluster_label in data_state.input_descriptor_samplings[0].index
        ]))]
        for output_0_descriptor_l in data_state.output_descriptor_samplings[0]
    ])
            

def compute_CHSM(data_state: DataStateSample) -> None:
    """Compute CHSM (Convex Hull Surface Modification) for each sampling and attach to the data state."""
    Cjs = data_state.index_dataframe.query("is_center == 1")["cluster"].value_counts().to_dict()
    rhoj = {}
    
    for cluster_label in data_state.input_descriptor_samplings[0].index:
        points_dfj = data_state.index_dataframe.query('cluster == @cluster_label')
        points_df0 = points_dfj.query('sampling_round == 0')
        root_set_indices = points_df0["root_set"].values
        rhoj[cluster_label] = {}
        
        for sampling_round in range(1, 1 + data_state.max_sampling_rounds):
            points_df = points_dfj.query('sampling_round == @sampling_round')
            rhoj[cluster_label][sampling_round] = [
                np.sqrt(np.sum((
                    data_state.viz0.embeddings[points_df0.index] 
                    - data_state.viz0.embeddings[[points_df.query("root_set == @root_set_i").index[0] for root_set_i in root_set_indices]]
                )**2) / Cjs[cluster_label]) + 1e-6
            ]

    data_state.CHSM_input = np.array([
        [
            mahalanobis(
                data_state.input_descriptor_samplings[0].loc[cluster_label, data_state.descriptors].values,
                data_state.input_descriptor_samplings[i].loc[cluster_label, data_state.descriptors].values,
                data_state.empirical_covariance_matrix_addI_inv
            ) / rhoj[cluster_label][i]
            for cluster_label in data_state.input_descriptor_samplings[0].index
        ]
        for i in range(1, 1 + data_state.max_sampling_rounds)
    ])

    data_state.CHSM_output = np.array([
        [
            [
                mahalanobis(
                    data_state.output_descriptor_samplings[0][l].loc[cluster_label, data_state.descriptors].values,
                    data_state.output_descriptor_samplings[i][l].loc[cluster_label, data_state.descriptors].values,
                    data_state.empirical_covariance_matrix_addI_inv
                ) / rhoj[cluster_label][i]
                for cluster_label in data_state.output_descriptor_samplings[0][l].index
            ]
            for l in range(len(data_state.LLMs))
        ]
        for i in range(1, 1 + data_state.max_sampling_rounds)
    ])
    
    
def compute_GRS(data_state: DataStateSample) -> None:
    """Compute GRS (Global Robustness Score) and attach to the data state."""
    points_df = data_state.index_dataframe.query("is_center == 1")
    data_state.GRS_input = GRS(data_state, points_df, data_state.input_descriptor_samplings[0].loc[:, data_state.descriptors], data_state.viz0)
    data_state.GRS_output = np.array([
        GRS(data_state, points_df, data_state.output_descriptor_samplings[0][l].loc[:, data_state.descriptors], data_state.llm_vizs[l]) 
        for l in range(len(data_state.LLMs))
    ])


def GRS(data_state: DataStateSample, points_df: pd.DataFrame, descriptor: pd.DataFrame, viz: JointViz) -> float:
    """Compute GRS metric relative to the provided visualizer."""
    wjs = data_state.index_dataframe.query("is_center == 1")["cluster"].value_counts(normalize=True).to_dict()
    descriptor_clustermean = np.sum([wjs[cluster_label] * descriptor.loc[cluster_label].values for cluster_label in wjs], axis=0)
    
    values = 0.0
    Z = 0.0
    
    for cluster_label, df in points_df.groupby('cluster'):
        values += wjs[cluster_label] * mahalanobis(descriptor.loc[cluster_label].values, descriptor_clustermean, data_state.empirical_covariance_matrix_addI_inv)
        Z += wjs[cluster_label] * euclidean_distances(viz.embeddings[df.index])[np.triu_indices(len(df), 1)].mean()
        
    return values / Z if Z else 0.0


def compute_shape_descriptor(data_state: DataStateSample, points_df: pd.DataFrame, viz: JointViz, num_eigenvalues: int = 100, eta: float = 1.0) -> pd.DataFrame:
    """Compute shape descriptors for a given set of points and a visualizer."""
    clusters = data_state.index_dataframe.loc[points_df.index, 'cluster']
    cluster_classes = list(set(clusters))
    descriptor = pd.DataFrame(index=cluster_classes)
    
    for cluster_label, df in tqdm(points_df.groupby('cluster'), desc="cluster shape desc"):
        descriptor.loc[cluster_label, 'Vol'] = ConvexHullVol_sample(viz.projections3D_euclidean, df.index)
        descriptor.loc[cluster_label, 'Area'] = ConvexHullVol_sample(viz.projections_euclidean, df.index)
        descriptor.loc[cluster_label, 'Peri'] = ConvexHullArea_sample(viz.projections_euclidean, df.index)
        descriptor.loc[cluster_label, 'GKDE'] = gaussian_kde_vol_sample(viz.projections_euclidean, df.index)
        
        D = euclidean_distances(viz.embeddings[df.index])[np.triu_indices(len(df), 1)]
        descriptor.loc[cluster_label, 'Dmean'] = D.mean()
        descriptor.loc[cluster_label, 'Dmax'] = D.max()
        
        descriptor.loc[cluster_label, [
            f'cov_lambda_{i+1}' for i in range(num_eigenvalues)
        ]] = empirical_covariance_eigenvalues(viz.embeddings[df.index], k=num_eigenvalues)

        descriptor.loc[cluster_label, [
            f'cor_lambda_{i+1}' for i in range(num_eigenvalues)
        ]] = empirical_correlation_eigenvalues(viz.embeddings[df.index], k=num_eigenvalues)
        
    descriptor = (descriptor + eta).apply(np.log)
    
    return descriptor
