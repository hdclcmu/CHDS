# CHDS: Geometry-Based Synthetic Data Generation for Tabular Data
## Requirements
- Python 3.12.4
- Numpy 2.4.6
- Pandas 3.0.3
- Scikit-learn 1.8.0
- Scipy 1.17.1
- Statsmodels 0.14.6
- Tqdm 4.67.3

## Usage

To generate synthetic data, define numerical columns, and pass original dataset to the `CHDS_synthesizer` function. 

Example:

```python
import pandas as pd
from CHDS import CHDS_synthesizer

# 1. Load your original dataset
df = pd.read_csv('your_dataset.csv')

# 2. Define numerical columns
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# 3. Generate synthetic data
synthetic_data = CHDS_synthesizer(
    input_df=df,
    num_records_to_generate=df.shape[0], # Generate the same number of rows as the original
    numerical_cols=numerical_cols,
    burn_in=2000
)

synthetic_data.head()
