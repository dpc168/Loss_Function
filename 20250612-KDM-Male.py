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
'''
np.random.seed(42)
X = np.random.randn(1000, 5)  # 输入特征 (1000, 5)
true_weights = np.array([1.5, -2.0, 3.0, 0.5, -1.0])  # 真实权重
y = X @ true_weights + np.random.randn(1000) * 0.1  # 目标值 (1000,)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
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
'''
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
feature_names = [...]  # 你的特征名称列表

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
'''

Sex=DATA1[1:,0]
Target=DATA1[1:,-1]

X_male= X[np.where(Sex == 1)[0],:]
Y_male= Y[np.where(Sex == 1)[0]]
Target_male=Target[np.where(Sex == 1)[0]]

X_male_h = X_male[np.where(Target_male == 0)[0],:]
Y_male_h = Y_male[np.where(Target_male == 0)[0]]
X_male_noh = X_male[np.where(Target_male == 1)[0],:]
Y_male_noh = Y_male[np.where(Target_male == 1)[0]]

'''
X_female= X[np.where(Sex == 0)[0],:]
Y_female= Y[np.where(Sex == 0)[0]]
Target_female=Target[np.where(Sex == 0)[0]]
'''

#★★★★★★★★★★★★★★★★★★★★★★★现在测试男性的数据
X_model=X_male_h
Y_model=Y_male_h
#print(np.size(Y_model))


X_train=X_model[::2]  # 从索引0开始，步长为2
y_train=Y_model[::2]

X_test=X_model[1::2]  # 从索引1开始，步长为2 
y_test=Y_model[1::2]


##★★★★★★★★★★★★★----KDM
# 修正后的KDM模型
def KDM_Age(X, Y, X_test, Y_test):
    KDM_age = np.zeros((1, len(Y_test)))
    table_kdm = np.zeros((3, X.shape[1]))

    for i in range(X.shape[1]):
        xx = X[:, i].reshape(-1, 1)
        Y1 = Y.reshape(-1, 1)
        model = LinearRegression()
        model.fit(Y1, xx)
        residuals = model.predict(Y1) - xx
        indices = [index for index, value in enumerate(residuals) if
                   -2.3 * np.std(residuals) < value < 2.3 * np.std(residuals)]

        xx_new = xx[indices]
        yy_new = Y1[indices]
        model.fit(yy_new, xx_new)
        xx_new_pre = model.predict(yy_new)

        # 修正数组到标量的转换问题
        table_kdm[0, i] = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
        table_kdm[1, i] = model.coef_[0, 0] if model.coef_.ndim > 1 else model.coef_[0]
        table_kdm[2, i] = np.sqrt(mean_squared_error(xx_new, xx_new_pre))

    model1 = LinearRegression()
    model1.fit(X, Y)
    Y_pred = model1.predict(X)
    Sba = np.sqrt(mean_squared_error(Y, Y_pred))
    print(f'KDM模型原始特征RMSE: {Sba:.4f}')

    for i in range(len(Y_test)):
        xxx = X_test[i, :]
        fenzi = np.dot((xxx - table_kdm[0, :]), (table_kdm[1, :] / (table_kdm[2, :] ** 2))) + Y_test[i] / (Sba ** 2)
        fenmu = np.dot(table_kdm[1, :] / table_kdm[2, :], table_kdm[1, :] / table_kdm[2, :]) + 1 / (Sba ** 2)
        KDM_age[0, i] = fenzi / fenmu

    return KDM_age

from sklearn.linear_model import LinearRegression
kdm_train=KDM_Age(X_train, y_train, X_train, y_train)
kdm_test =KDM_Age(X_train, y_train, X_test,  y_test)


plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
plt.subplot(2, 3, 1)  # (行数, 列数, 当前子图索引)
plt.scatter(y_train, kdm_train,s=10,color='b', alpha=0.2)
#mean_squared_error(y_train, train_predictions)
corr_coef = np.corrcoef(np.array(y_train, dtype=float), kdm_train.ravel() )[0, 1]
''''''
modela = LinearRegression()
modela.fit(y_train.reshape(-1, 1), kdm_train.reshape(-1, 1))
predia = modela.predict(np.array([25,85]).reshape(-1, 1))
plt.plot([25,85], predia, color='red', linewidth=1.5, linestyle='-')
a = modela.coef_[0].item()  # 斜率
b = modela.intercept_.item()  # 截距
equation = f'y = {a:.2f}x + {b:.2f}\nr = {corr_coef:.2f}'
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
         fontsize=13, color='red', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
plt.xlim(20, 90)
plt.ylim(20, 90)
plt.plot([20,90], [20,90], color='black', linewidth=1, linestyle='--', label='y=x')
#plt.plot([20,90], [20,90], color='red', linewidth=1, linestyle='-', label='y=x')
plt.xlabel('Chronological age')
plt.ylabel('Pre-Arterial biological age')
plt.title(f'Male (KDM, train, n={len(y_train)})')

plt.subplot(2, 3, 2)  # (行数, 列数, 当前子图索引)
plt.scatter(y_train, kdm_train-y_train,s=10,color='b', alpha=0.2)
modela.fit(y_train.reshape(-1, 1), kdm_train.reshape(-1, 1)-y_train.reshape(-1, 1))
predia = modela.predict(np.array([25,85]).reshape(-1, 1))
#plt.plot([25,85], predia, color='red', linewidth=1.5, linestyle='-')
plt.xlabel('Chronological age')
plt.ylabel('Pre-Residual')
plt.xlim(20, 90)
plt.ylim(-35, 35)
plt.title(f'Male (KDM, train, n={len(y_train)})')

plt.subplot(2, 3, 3)  # (行数, 列数, 当前子图索引)
sns.set(style='whitegrid')
sns.distplot(kdm_train-y_train, kde=True, bins=20, color='b', hist_kws={'edgecolor':'black'})
plt.ylabel('Probability')
plt.xlabel('Pre-Residual')
plt.xlim(-35, 35)
plt.ylim(0, 0.075)
plt.title(f'Male (KDM, train, n={len(y_train)})')

plt.subplot(2, 3, 4)  # (行数, 列数, 当前子图索引)
plt.scatter(y_test, kdm_test,s=10,color='c', alpha=0.2)
corr_coef = np.corrcoef(np.array(y_test, dtype=float), kdm_test.ravel() )[0, 1]
modela.fit(y_test.reshape(-1, 1), kdm_test.reshape(-1, 1))
predia = modela.predict(np.array([25,85]).reshape(-1, 1))
plt.plot([25,85], predia, color='red', linewidth=1.5, linestyle='-')
a = modela.coef_[0].item()  # 斜率
b = modela.intercept_.item()  # 截距
equation = f'y = {a:.2f}x + {b:.2f}\nr = {corr_coef:.2f}'
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
         fontsize=13, color='red', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
plt.xlim(20, 90)
plt.ylim(20, 90)
plt.plot([20,90], [20,90], color='black', linewidth=1, linestyle='--', label='y=x')
#plt.plot([20,90], [20,90], color='red', linewidth=1, linestyle='-', label='y=x')
plt.xlabel('Chronological age')
plt.ylabel('Pre-Arterial biological age')
plt.title(f'Male (KDM, test, n={len(y_test)})')

plt.subplot(2, 3, 5)  # (行数, 列数, 当前子图索引)
plt.scatter(y_test, kdm_test-y_test,s=10,color='c', alpha=0.2)
modela.fit(y_test.reshape(-1, 1), kdm_test.reshape(-1, 1)-y_test.reshape(-1, 1))
predia = modela.predict(np.array([25,85]).reshape(-1, 1))
#plt.plot([25,85], predia, color='red', linewidth=1.5, linestyle='-')
plt.xlim(20, 90)
plt.xlabel('Chronological age')
plt.ylabel('Pre-Residual')
plt.xlim(20, 90)
plt.ylim(-35, 35)
plt.title(f'Male (KDM, test, n={len(y_test)})')

plt.subplot(2, 3, 6)  # (行数, 列数, 当前子图索引)
sns.set(style='whitegrid')
sns.distplot(kdm_test-y_test, kde=True, bins=20, color='c', hist_kws={'edgecolor':'black'})
plt.ylabel('Probability')
plt.xlabel('Pre-Residual')
plt.xlim(-35, 35)
plt.ylim(0, 0.075)
#plt.title('Male (MLR, test)')
plt.title(f'Male (KDM, test, n={len(y_test)})')
plt.tight_layout()
plt.show()


MAE=mean_absolute_error(y_test, kdm_test.flatten())################################MAE--5.161
MSE=mean_squared_error(y_test, kdm_test.flatten())#train_mse#######################MSE--40.88
RMSE=MSE**0.5
r=np.corrcoef(np.array(y_test, dtype=float), kdm_test.ravel() )[0, 1]
R2=r2_score(y_test, kdm_test.flatten())###########################################R2=0.568
    #corr1, p_value1 = spearmanr(y_train, re_train_predictions-y_train)
    #bianhua[4,i]=corr1
test_result = kdm_test.flatten()-y_test
    # 找到小于50的样本的索引
indices_mi = np.where(y_test.astype(int) < 50)[0]
    #len(indices_mi)
    # 计算大于0的样本数量
    #np.sum(train_result[indices_mi].astype(float) > 0)
P1=np.sum(test_result[indices_mi].astype(float) > 0)/len(indices_mi)#############小于50--0.51
    # 找到大于等于50的样本的索引
indices_ma = np.where(y_test.astype(int) >= 50)[0]
    #len(indices_mi)
    # 计算大于0的样本数量
    #np.sum(train_result[indices_mi].astype(float) > 0)
P2=np.sum(test_result[indices_ma].astype(float) > 0)/len(indices_ma)##############大于50---0.52

print(MAE,MSE,RMSE,r,R2,P1,P2)


#############     混淆矩阵分析
kdm_male =KDM_Age(X_train, y_train, X_male,Y_male)
fenxi_=kdm_male-Y_male
from sklearn.metrics import confusion_matrix
#y_true = [1, 0, 1, 1, 0, 1, 0, 0]
#y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
#cm = confusion_matrix(y_true, y_pred)
#Target_male.shape
#(fenxi_ > 0).astype(int).reshape(-1).shape
cm = confusion_matrix(Target_male.astype(int), (fenxi_ > 0).astype(int).reshape(-1))

# 提取混淆矩阵元素
TN, FP = cm[0,0], cm[0,1]
FN, TP = cm[1,0], cm[1,1]

# 计算指标
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

# Sigmoid 转换
sigmoid_arr = 1 / (1 + np.exp(-fenxi_.astype(np.float64)))
#plt.plot(sigmoid_arr.flatten(),sigmoid_arr.flatten(), color='red', linewidth=1.5, linestyle='-')
# 转换数据格式
probabilities = sigmoid_arr.flatten()
labels = Target_male.astype(np.int32)
# 计算 AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labels, probabilities)

print(accuracy,precision,recall,f1,auc)



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# 假设已有 sigmoid_arr 和 Target_male 数据
# sigmoid_arr = 1 / (1 + np.exp(-fenxi_.astype(np.float64)))
# probabilities = sigmoid_arr.flatten()
# labels = Target_male.astype(np.int32)

# 计算ROC曲线的关键指标
fpr, tpr, thresholds = roc_curve(labels, probabilities)
roc_auc = auc(fpr, tpr)  # 或使用之前计算的 auc 值

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

