from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# 使用sk-learn实现多项式拟合

# 自动生成多项式特征矩阵PolynomialFeatures
# X=[2,-1,3]
# X_reshape = np.array(X).reshape(len(X),1) #转换为向量
# print(PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_reshape))

x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]
y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]
x = np.array(x).reshape(len(x),1)
y = np.array(y).reshape(len(y),1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_x = poly_features.fit_transform(x)
# print(poly_x)
#训练线性回归模型
model = LinearRegression()
model.fit(poly_x, y)

# print(model.intercept_, model.coef_)
# 绘制拟合图形
x_temp = np.linspace(0, 80, 10000)
x_temp = np.array(x_temp).reshape(len(x_temp), 1)
poly_x_temp = poly_features.fit_transform(x_temp) # 将临时点变为二次项的样子

plt.plot(x_temp, model.predict(poly_x_temp), 'r')
plt.scatter(x, y)
plt.show()

