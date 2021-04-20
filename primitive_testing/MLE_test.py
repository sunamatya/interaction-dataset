import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats
import pymc3 as pm3
import numdifftools as ndt
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

if __name__ == "__main__":
    N = 100
    x = np.linespace(0,20,N)
    e = np.random.normal(loc= 0.0, scale= 5.0, size= N)
    y = 3*x+ e
    df = pd.DataFrame({'y':y, 'x':x})
    df['constant'] = 1

    #plot
    sns.regplot(df.x, df.y)
