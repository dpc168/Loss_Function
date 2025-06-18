import pandas as pd
import matplotlib.pyplot as plt

DATA_MLR = pd.read_excel('C:/Users/32684/.spyder-py3/20250418自定义损失/MLR_Male.xlsx', header=None)
DATA_MLR = DATA_MLR.values
print(DATA_MLR)
X=DATA_MLR[0,1:]

DATA_RF = pd.read_excel('C:/Users/32684/.spyder-py3/20250418自定义损失/RF_Male.xlsx', header=None)
DATA_RF = DATA_RF.values

DATA_XGBoost = pd.read_excel('C:/Users/32684/.spyder-py3/20250418自定义损失/XGBoost_Male.xlsx', header=None)
DATA_XGBoost = DATA_XGBoost.values


################对陆慕他变化，进行可视化
plt.figure(figsize=(20, 13))  # 调整整体画布大小
plt.subplot(3, 4, 1)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[1,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[1,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[1,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [4.56,4.56], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [4,16], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [4,16], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(4, 16)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('MAE', fontsize=16)
plt.title('Male (test, n=6060)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={'size': 15})

#plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
plt.subplot(3, 4, 2)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[2,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[2,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[2,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [33.12,33.12], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [20,220], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [20,220], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(20, 220)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.title('Male (test, n=6060)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={'size': 15})


#plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
#0.834
plt.subplot(3, 4, 3)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[3,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[3,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[3,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [5.75,5.75], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [5,20], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [5,20], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(5, 20)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.title('Male (test, n=6060)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={'size': 15})


#plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
#R2=0.568
plt.subplot(3, 4, 4)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[4,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[4,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[4,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.88,0.88], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [-0.6,1], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [-0.6,1], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(-0.6, 1)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('r', fontsize=16)
plt.title('Male (test, n=6060)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower right', bbox_to_anchor=(1, 0),prop={'size': 15})


#plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
#0.51
plt.subplot(3, 4, 5)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[5,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[5,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[5,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.69,0.69], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [-1,0.8], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [-1,0.8], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(-1, 0.8)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel(r'$\mathrm{R}^2$', fontsize=16)
plt.title('Male (test, n=6060)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper left', bbox_to_anchor=(0, 1),prop={'size': 15})


#plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
plt.subplot(3, 4, 6)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[6,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[6,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[6,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.49,0.49], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.4,1], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.4,1], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.4, 1)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('P1', fontsize=16)
plt.title('Male (test, n=6060)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper left', bbox_to_anchor=(0, 1),prop={'size': 15})
#plt.subplots_adjust(wspace=0.4)  # 增加水平间距（默认0.2）


plt.subplot(3, 4, 7)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[7,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[7,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[7,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.51,0.51], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0,0.6], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0,0.6], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0, 0.6)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('P2', fontsize=16)
plt.title('Male (test, n=6060)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={'size': 15})

plt.subplot(3, 4, 8)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[8,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[8,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[8,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.73,0.73], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.2,0.8], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.2,0.8], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.2, 0.8)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Male (n=22959)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),prop={'size': 15})


plt.subplot(3, 4, 9)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[9,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[9,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[9,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.64,0.64], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.1, 0.7], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.1, 0.7], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.1, 0.7)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.title('Male (n=22959)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),prop={'size': 15})


plt.subplot(3, 4, 10)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[10,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[10,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[10,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.98,0.98], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.2, 1.05], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.2, 1.05], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.2, 1.05)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.title('Male (n=22959)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),prop={'size': 15})

plt.subplot(3, 4, 11)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[11,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[11,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[11,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.77,0.77], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.2, 0.8], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.2, 0.8], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.2, 0.8)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('F1-Score', fontsize=16)
plt.title('Male (n=22959)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),prop={'size': 15})

plt.subplot(3, 4, 12)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[12,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[12,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[12,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.94,0.94], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.2, 1.0], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.2, 1.0], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.2, 1.0)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('AUC', fontsize=16)
plt.title('Male (n=22959)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),prop={'size': 15})

# After creating all subplots, adjust the layout
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust horizontal and vertical spacing
# Or use tight_layout which often works well
#plt.tight_layout(pad=2.0)  # Add some padding around the figure
plt.tight_layout()  # 自动调整子图间距
plt.show()

#----------------------------------以下是女性
DATA_MLR = pd.read_excel('C:/Users/32684/.spyder-py3/20250418自定义损失/MLR_Female.xlsx', header=None)
DATA_MLR = DATA_MLR.values
print(DATA_MLR)
X=DATA_MLR[0,1:]

DATA_RF = pd.read_excel('C:/Users/32684/.spyder-py3/20250418自定义损失/RF_Female.xlsx', header=None)
DATA_RF = DATA_RF.values

DATA_XGBoost = pd.read_excel('C:/Users/32684/.spyder-py3/20250418自定义损失/XGBoost_Female.xlsx', header=None)
DATA_XGBoost = DATA_XGBoost.values


################对陆慕他变化，进行可视化
plt.figure(figsize=(20, 13))  # 调整整体画布大小
plt.subplot(3, 4, 1)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[1,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[1,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[1,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [5.16,5.16], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [4,16], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [4,16], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(4, 16)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('MAE', fontsize=16)
plt.title('Female (test, n=4368)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={'size': 15})

#plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
plt.subplot(3, 4, 2)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[2,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[2,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[2,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [40.88,40.88], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [20,220], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [20,220], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(20, 220)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.title('Female (test, n=4368)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={'size': 15})


#plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
#0.834
plt.subplot(3, 4, 3)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[3,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[3,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[3,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [6.39,6.39], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [5,20], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [5,20], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(5, 20)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.title('Female (test, n=4368)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={'size': 15})


#plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
#R2=0.568
plt.subplot(3, 4, 4)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[4,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[4,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[4,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.83,0.83], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [-0.6,1], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [-0.6,1], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(-0.6, 1)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('r', fontsize=16)
plt.title('Female (test, n=4368)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower right', bbox_to_anchor=(1, 0),prop={'size': 15})


#plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
#0.51
plt.subplot(3, 4, 5)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[5,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[5,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[5,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.57,0.57], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [-1,0.8], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [-1,0.8], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(-1, 0.8)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel(r'$\mathrm{R}^2$', fontsize=16)
plt.title('Female (test, n=4368)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper left', bbox_to_anchor=(0, 1),prop={'size': 15})


#plt.figure(figsize=(8.8, 6))  # 调整整体画布大小
plt.subplot(3, 4, 6)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[6,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[6,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[6,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.51,0.51], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.4,1], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.4,1], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.4, 1)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('P1', fontsize=16)
plt.title('Female (test, n=4368)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper left', bbox_to_anchor=(0, 1),prop={'size': 15})
#plt.subplots_adjust(wspace=0.4)  # 增加水平间距（默认0.2）


plt.subplot(3, 4, 7)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[7,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[7,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[7,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.52,0.52], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0,0.6], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0,0.6], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0, 0.6)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('P2', fontsize=16)
plt.title('Female (test, n=4368)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={'size': 15})

plt.subplot(3, 4, 8)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[8,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[8,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[8,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.66,0.66], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.2,0.8], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.2,0.8], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.2, 0.8)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Female (n=13264)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),prop={'size': 15})


plt.subplot(3, 4, 9)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[9,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[9,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[9,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.5,0.5], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.1, 0.7], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.1, 0.7], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.1, 0.7)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.title('Female (n=13264)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),prop={'size': 15})


plt.subplot(3, 4, 10)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[10,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[10,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[10,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.98,0.98], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.2, 1.05], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.2, 1.05], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.2, 1.05)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.title('Female (n=13264)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),prop={'size': 15})

plt.subplot(3, 4, 11)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[11,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[11,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[11,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.77,0.77], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.2, 0.8], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.2, 0.8], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.2, 0.8)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('F1-Score', fontsize=16)
plt.title('Female (n=13264)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),prop={'size': 15})

plt.subplot(3, 4, 12)  # (行数, 列数, 当前子图索引)
plt.scatter(X,DATA_MLR[12,1:],s=25,label='MLR', marker='o')
plt.scatter(X,DATA_RF[12,1:],s=25,label='RF', marker='^')
plt.scatter(X,DATA_XGBoost[12,1:],s=25,label='XGBoost', marker='s')
plt.plot([-2,2], [0.94,0.94], color='black', linewidth=3.5, linestyle='--', label='KDM')
plt.plot([1,1], [0.2, 1.0], color='red', linewidth=3.5, linestyle=':', label='Loss$_{\mathrm{(MSE)}}$')
plt.plot([0,0], [0.2, 1.0], color='red', linewidth=3.5, linestyle='-', label='Loss1')
plt.xlim(-2, 2)
plt.ylim(0.2, 1.0)
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel('AUC', fontsize=16)
plt.title('Female (n=13264)', fontsize=18)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], rotation=45, fontsize=16)  # 指定要显示的刻度值
plt.yticks(fontsize=16)  # Y轴刻度
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),prop={'size': 15})

# After creating all subplots, adjust the layout
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust horizontal and vertical spacing
# Or use tight_layout which often works well
#plt.tight_layout(pad=2.0)  # Add some padding around the figure
plt.tight_layout()  # 自动调整子图间距
plt.show()