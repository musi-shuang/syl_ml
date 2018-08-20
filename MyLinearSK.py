'''scikit-learn 线性回归拟合'''
import numpy as np
from sklearn.linear_model import LinearRegression

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