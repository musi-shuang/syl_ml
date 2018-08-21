'''scikit-learn 线性回归拟合'''
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

# 定义线性回归模型
# 构造函数中的默认的四个参数含义如下：
# fit_intercept=True 是否计算截距
# normalize=False 是否对数据进行标准化
# copy_X=True 对数据的副本进行操作
# n_jobs=1 作业数量，若为-1，则为所有的CPU参与运算
model = LinearRegression()
#将x自变量转置为单列矩阵，后模型训练（fit）
print(x.reshape(len(x), 1))
model.fit(x.reshape(len(x), 1), y)

#获取模型参数，intercept_为截距，coef_为x系数 与python原生算法一致，此时得到的参数是一个矩阵
print(model.intercept_, model.coef_)
#预测，输入值应为一个矩阵，返回y值
print(model.predict([[150]])) # 同下
print(model._decision_function([[150]]))

print(model.score(x.reshape(len(x), 1),y)) #评分函数，返回一个小于或等于1的值

print(model.get_params(True)) #{'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False}


#遗留问题，最小二乘法的算法复杂度。
# 假设影响因素 x 为一个 n 行 p 列的矩阵那么其算法复杂度为O(np^2) 假设n >= p


# 使用矩阵进行并行化运算
x = np.matrix([[1,56], [1,72], [1,69], [1,88], [1,102], [1,86],[1, 76], [1,79], [1,94], [1,74]])
y = np.matrix([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

model = LinearRegression()
model.fit(x, y.T)  # 转置列向量
print(model.intercept_, model.coef_)


# 波士顿房价数据集处理
df = pd.read_csv("D:\py\course-5-boston.csv")
# print(df.head()) # 展示前5行数据
features = df[['crim','rm','lstat']]
# print(features.describe()) # 统计了上述字段的一些值：个数、平均值、最大值、最小值

# 将样本分为训练集和测试集
target = df['medv']  # 城镇住房价格中位数
split_num = int(len(features)*0.7) #得到样本70%的位置
train_x = features[:split_num] # 分片，获取训练集
train_y = target[:split_num]

test_x = features[split_num:] # 获取测试机
test_y = target[split_num:]

model = LinearRegression()
model.fit(train_x, train_y)
# print(model.coef_, model.intercept_)
preds = model.predict(test_x) # 模型对测试集的预测结果
# 计算绝对误差 python
def mae_value(y_true, y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true-y_pred))/n
    return mae

# 计算均方误差，表示误差的平方的期望值
def mse_value(y_true, y_pred):
    n = len(y_true)
    mse = sum(np.square(y_true - y_pred))/n
    return mse

mae = mae_value(test_y.values, preds)
mse = mse_value(test_y.values, preds)
print("MAE", mae)
print("MSE", mse)
# 结果
# MAE 13.02206307278031
# MSE 303.8331247223652