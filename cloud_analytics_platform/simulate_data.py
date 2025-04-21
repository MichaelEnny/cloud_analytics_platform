import pandas as pd
import numpy as np

def generate_hospital_data(n=1000):
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 90, n),
        'chronic_disease': np.random.randint(0, 2, n),
        'prev_admissions': np.random.poisson(2, n),
        'readmitted': np.random.binomial(1, 0.3, n)
    })
    return df

def generate_credit_data(n=1000):
    np.random.seed(42)
    df = pd.DataFrame({
        'income': np.random.normal(50000, 15000, n),
        'credit_score': np.random.randint(300, 850, n),
        'loan_amount': np.random.normal(15000, 5000, n),
        'defaulted': np.random.binomial(1, 0.2, n)
    })
    return df
