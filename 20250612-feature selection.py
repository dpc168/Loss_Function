import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error  # 正确导入方式
from sklearn.metrics import r2_score  # 必须导入！
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from scipy.stats import pearsonr

# 11111-----------------------生成数据 (1000个样本，5个特征)
DATA = pd.read_excel('C:/Users/32684/.spyder-py3/20250418自定义损失/Atherosclerosis_data.xlsx', header=None)

DATA1 = DATA.values
print(DATA1)

Y=DATA1[1:,1]
print(Y)

#X=DATA1[1:,[3,4]]
X1=DATA1[1:,2:-1]
print(X1)

print(DATA1[0,2:-1])

X=X1[:,[0,1,2,4,7]]


from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np

# 假设 X 是特征矩阵，y 是年龄
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X1)  # 标准化特征

# 使用交叉验证选择最优 lambda
lasso = LassoCV(cv=5).fit(X_scaled, Y)

# 特征重要性：系数绝对值
importance = np.abs(lasso.coef_)
#feature_names = [...]  # 你的特征名称列表

##★★★★★★★★★★★★★★★★★★★★★置信区间
from sklearn.utils import resample

def bootstrap_ci(X, y, model, n_bootstraps=1000, alpha=0.05):
    coefs = []
    for _ in range(n_bootstraps):
        X_resampled, y_resampled = resample(X, y)
        model.fit(X_resampled, y_resampled)
        coefs.append(model.coef_)
    
    coefs = np.array(coefs)
    lower = np.percentile(coefs, (alpha/2)*100, axis=0)
    upper = np.percentile(coefs, (1-alpha/2)*100, axis=0)
    return lower, upper

# 计算95%置信区间
lower_ci, upper_ci = bootstrap_ci(X_scaled, Y, lasso)

##★★★★★★★★★★★★★★★★★★★★★p值
from sklearn.base import clone

def permutation_test(X, y, model, n_permutations=1000):
    observed_coef = model.fit(X, y).coef_
    perm_coefs = np.zeros((n_permutations, X.shape[1]))
    
    for i in range(n_permutations):
        y_permuted = np.random.permutation(y)
        perm_model = clone(model)
        perm_coefs[i] = perm_model.fit(X, y_permuted).coef_
    
    p_values = np.mean(np.abs(perm_coefs) >= np.abs(observed_coef), axis=0)
    return p_values

# 计算p值
p_values = permutation_test(X_scaled, Y, lasso)
