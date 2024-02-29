import numpy as np
import pandas as pd
import warnings
from numba import njit

from tqdm import tqdm
from typing import Optional, Union

def default(x, default):
    return default if x is None else x

@njit
def stationary_bootstrap(n: int, n_boot: int, block_len: int):
    p = 1.0 / block_len
    for _ in range(n_boot):
        indices = np.zeros(n, dtype=np.int32)
        indices[0] = int(np.random.random() * n)
        select = np.random.random(n) < p
        indices[select] = (np.random.random(select.sum()) * n).astype(np.int32)
        for t in range(1, n):
            if not select[t]:
                indices[t] = indices[t - 1] + 1
        # Wrap around
        indices[indices > n - 1] -= n
        yield indices

@njit
def block_bootstrap(n: int, n_boot: int, block_len: int):
    for _ in range(n_boot):
        indices = np.zeros(n, dtype=np.int32)
        indices[0] = int(np.random.random() * n)
        counter = 0
        for t in range(1, n):
            counter = (counter + 1) % block_len
            if counter == 0:
                indices[t] = int(np.random.random() * n)
            else:
                indices[t] = indices[t - 1] + 1
        # Wrap around
        indices[indices > n - 1] -= n
        yield indices

def pval_R(z: np.ndarray, z_data: np.ndarray) -> float:
    TR_dist = np.abs(z).max(axis=(1, 2))
    TR = z_data.max()
    return (TR_dist > TR).mean()

def pval_SQ(z: np.ndarray, z_data: np.ndarray) -> float:
    dist = (z ** 2).sum(axis=(1, 2)) / 2
    return (dist > ((z_data ** 2).sum() / 2)).mean()

class ModelConfidenceSet:
    """
    A class for conducting the Model Confidence Set (MCS) procedure by Hansen, Lunde,
    and Nason (2011), which evaluates and compares the performance of multiple 
    predictive models based on their loss functions. The MCS provides a set of models 
    that are statistically indistinguishable from the best model at a given 
    confidence level.

    The MCS method is a statistical tool used for model selection and comparison,
    offering a way to identify a subset of models that are not significantly worse
    than the best model, according to their predictive accuracy or loss.

    Parameters
    ----------
    losses : np.ndarray or pd.DataFrame
        An array or DataFrame of shape (n_samples, n_models) containing the loss values
        associated with each model. Each column represents a model, and each row represents
        a loss value for a particular instance.
    n_boot : int, optional
        The number of bootstrap replications to use in the MCS procedure. Default is 5000.
    alpha : float, optional
        The significance level for determining the confidence set. Must be between 0 and 1.
        Default is 0.05.
    block_len : int, optional
        The length of blocks to use in block bootstrap methods. If None, it defaults to
        the number of observations. Only applicable if bootstrap_variant is "block".
    bootstrap_variant : {'stationary', 'block'}, optional
        The type of bootstrap to use. "stationary" for stationary bootstrap, and "block"
        for block bootstrap. Default is "stationary".
    method : {'R', 'SQ'}, optional
        The method to compute p-values. "R" for the Romano-Wolf method, and "SQ" for
        the sequential method. Default is "R".
    show_progress : bool, optional
        If True, shows a progress bar during the bootstrap computation and MCS procedure.
        Default is False.

    Attributes
    ----------
    included : np.ndarray
        An array of model indices that are included in the model confidence set at the
        specified alpha level.
    excluded : np.ndarray
        An array of model indices that are excluded from the model confidence set.
    pvalues : np.ndarray
        The p-values associated with each model, used to determine inclusion in the MCS.
    results : dict
        A dictionary containing the p-values and status (included/excluded) for 
        each model. If compute() has not been called, results will be None.

    Methods
    -------
    compute():
        Executes the MCS procedure, computing the set of models that are statistically
        indistinguishable from the best model.
    results() -> pd.DataFrame:
        Returns a DataFrame with the p-values and status (included/excluded) for each
        model, indexed by model names. If compute() has not been called, it will be
        executed before returning the results. If as_dataframe is False, 
        returns a dictionary.

    Examples
    --------
    >>> import numpy as np
    >>> losses = np.random.rand(100, 3)  # Simulated loss data for 3 models over 100 samples
    >>> mcs = ModelConfidenceSet(losses, n_boot=1000, alpha=0.05, show_progress=True)
    >>> mcs.compute()
    >>> results = mcs.results()
    >>> print(results)

    Notes
    -----
    The MCS procedure assumes that lower loss values indicate better model performance.
    Ensure that your loss function is consistent with this assumption.
    """

    def __init__(self, losses: np.ndarray, n_boot: int=5_000, 
                 alpha: float=0.05, block_len: Optional[int]=None, 
                 bootstrap_variant: str="stationary", method: str="R",
                 show_progress: bool=False) -> None:
    
        # Input validation
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        if losses.ndim != 2:
            raise ValueError("losses must be 2-dimensional")
        if losses.shape[1] <= 1:
            raise ValueError("losses must have more than one column (models)")
        if n_boot <= 0:
            raise ValueError("n_boot must be positive")
        block_len = default(block_len, losses.shape[0])
        if block_len <= 0:
            raise ValueError("block_len must be positive")
        if block_len > losses.shape[0]:
            raise ValueError("block_len must be smaller than the number of rows")
        if bootstrap_variant not in ("stationary", "block"):
            raise ValueError("bootstrap_variant must be one of 'stationary' or 'block'")
        if method not in ("R", "SQ"):
            raise ValueError("method must be one of 'R' or 'SQ'")
        if n_boot < 1_000:
            warnings.warn("n_boot is less than 1,000, which may lead to inaccurate results")
        if not isinstance(show_progress, bool):
            raise ValueError("show_progress must be a boolean")
        
        if isinstance(losses, pd.DataFrame):
            self.model_names = losses.columns
            self.losses = losses.values
        else:
            self.model_names = np.arange(1, losses.shape[1] + 1)
            self.losses = losses
        self.alpha = alpha
        self.n_boot = n_boot
        self.block_len = block_len
        self.bootstrap = stationary_bootstrap if bootstrap_variant == "stationary" else block_bootstrap
        self.bootstrap(2, 1, 1) # Run once for numba compilation
        self.pval_fn = pval_R if method == "R" else pval_SQ
        self.show_progress = show_progress
        
        self.included = None
        self.excluded = None
        self.pvalues = None

    def compute(self) -> None:
        n, m0 = self.losses.shape
        mloss = self.losses.mean(axis=0, keepdims=True)
        dij_bar = mloss - mloss.T

        # Run bootstrap
        dij_bar_boot = np.zeros((self.n_boot, m0, m0))
        itr = enumerate(block_bootstrap(n, self.n_boot, self.block_len))
        if self.show_progress:
            itr = tqdm(itr, total=self.n_boot, desc="Bootstrapping")

        for i, bi in itr:
            bloss = self.losses[bi, :].mean(axis=0, keepdims=True)
            dij_bar_boot[i, ...] = bloss - bloss.T

        # Compute bootstrapped standard deviation estimates
        dij_std = np.sqrt(np.mean((dij_bar_boot - dij_bar) ** 2, axis=0) + np.eye(m0))

        z0 = (dij_bar_boot - dij_bar) / dij_std
        z0_data = dij_bar / dij_std

        excluded = np.zeros(m0)
        pvals = np.ones(m0)
        models = np.arange(1, m0 + 1)

        itr = range(m0 - 1)
        if self.show_progress:
            itr = tqdm(itr, desc="Computing MCS", unit="model")

        for i in itr:
            included = np.setdiff1d(models, excluded) - 1
            m = len(included)
            scale = m / (m - 1)
            pvals[i] = self.pval_fn(
                z0[:, *np.ix_(included, included)], 
                z0_data[np.ix_(included, included)])
            
            # Compute model to remove
            di_bar = np.mean(dij_bar[np.ix_(included, included)], axis=0) * scale
            di_bar_boot = dij_bar_boot[:, *np.ix_(included, included)].mean(axis=1) * scale
            di_std = np.sqrt(np.mean((di_bar_boot - di_bar) ** 2, axis=0))
            t = di_bar / di_std
            excluded[i] = included[np.argmax(t)] + 1

        # MCS p-values are the max up to that point
        pvals = np.maximum.accumulate(pvals)
        # Add last model
        excluded[-1] = np.setdiff1d(models, excluded)[-1]

        # Included models are where pval >= alpha
        self.included = excluded[pvals >= self.alpha]
        self.excluded = excluded[pvals < self.alpha]
        self.pvalues = pvals

    def results(self, as_dataframe=True) -> Union[dict, pd.DataFrame]:
        if self.included is None:
            self.compute()
        
        idx = np.concatenate([self.excluded, self.included]).astype(int) - 1

        self.results = {
            "pvalues": self.pvalues,
            "status": np.where(self.pvalues >= self.alpha, "included", "excluded"),
            "models": self.model_names[idx]
        }
        if as_dataframe:
            df = pd.DataFrame(self.results)
            df.index = df.pop("models")
            return df
        else:
            return self.results