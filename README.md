# Model Confidence Set

The `model-confidence-set` package provides a Python implementation of the Model Confidence Set (MCS) procedure [(Hansen, Lunde, and Nason, 2011)](https://www.jstor.org/stable/41057463), a statistical method for comparing and selecting models based on their performance. It allows users to identify a set of models that are statistically indistinguishable from the best model, given a statistical confidence level.

This package
- supports both stationary and block bootstrap methods.
- implements two methods for p-value computation: `R` and `SQ`.
- optionally displays progress during computation.

## Installation

To install `model-confidence-set`, simply use pip:

```bash
pip install model-confidence-set
```

## Usage
To use the Model Confidence Set in your Python code, follow the example below:

```python
import numpy as np
import pandas as pd
from model_confidence_set import ModelConfidenceSet

# Example losses matrix where rows are observations and columns are models
losses = np.random.rand(100, 5)  # 100 observations for 5 models

# Initialize the MCS procedure (5'000 bootstrap iterations, 5% confidence level)
mcs = ModelConfidenceSet(losses, n_boot=5000, alpha=0.05, show_progress=True)

# Compute the MCS
mcs.compute()

# Retrieve the results as a pandas DataFrame (use as_dataframe=False for a dict)
results = mcs.results()
print(results)
```

## Parameters
- `losses`: A 2D `numpy.ndarray` or `pandas.DataFrame` containing loss values of models. Rows correspond to observations, and columns correspond to different models.
- `n_boot`: Number of bootstrap replications for computing p-values. Default is `5000`.
- `alpha`: Significance level for determining model confidence set. Default is `0.05`.
- `block_len`: The length of blocks for the block bootstrap. If `None`, it defaults to the number of observations.
- `bootstrap_variant`: Specifies the bootstrap variant to use. Options are `'stationary'` or `'block'`. Default is `'stationary'`.
- `method`: The method used for p-value calculation. Options are `'R'` or `'SQ'`. Default is `'R'`.
- `show_progress`: Whether to show a progress bar during bootstrap computations. Default is `False`.

## Acknowledgments
This package draws inspiration from 
+ the [Matlab implementation by Kevin Sheppard](https://www.kevinsheppard.com/code/matlab/mfe-toolbox/) 
+ the [Python implementation by Michael Gong](https://michael-gong.com/blogs/model-confidence-set/).