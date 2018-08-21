import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


df = pd.read_csv("D:\py\course-6-vaccine.csv")
x = df['Year']
y = df['Values']
plt.plot(x,y,'r')
plt.scatter(x,y)
plt.show()  # 根据绘图结果，猜测多项式回归优于线性回归

# 划分数据为训练集与测试集
train_df = df[:int(len(df)*0.7)]
test_df = df[int(len(df)*0.7):]

train_x = train_df['Year'].values
train_y = train_df['Values'].values

test_x = test_df['Year'].values
test_y = test_df['Values'].values

# 线性回归
model = LinearRegression()
model.fit(train_x.reshape(len(train_x),1),train_y.reshape(len(train_y),1))
results = model.predict(test_x.reshape(len(test_x),1))
# print(results)
# 平均绝对误差和均方差
print("线性回归平均绝对误差：",mean_absolute_error(test_y, results.flatten()))
print("线性回归均方误差：",mean_squared_error(test_y, results.flatten()))

# 二次多项式预测
poly_features_2 = PolynomialFeatures(degree=2,include_bias=False)
poly_train_x_2 = poly_features_2.fit_transform(train_x.reshape(len(train_x),1))
poly_test_x_2 = poly_features_2.fit_transform(test_x.reshape(len(test_x),1))
model2 = LinearRegression()
model2.fit(poly_train_x_2, train_y.reshape(len(train_x),1))
results2 = model2.predict(poly_test_x_2)

print("二次多项式回归平均绝对误差：",mean_absolute_error(test_y, results2.flatten()))
print("二次多项式回归均方误差：",mean_squared_error(test_y, results2.flatten()))
# 得出结论：线性回归优于而次多项式回归

#使用make_pipeline管道类一次性实现多项式拟合
train_x = train_x.reshape(len(train_x),1)
test_x = test_x.reshape(len(test_x),1)
train_y = train_y.reshape(len(train_y),1)
for m in [3,4,5]:
    model3 = make_pipeline(PolynomialFeatures(m, include_bias=False), LinearRegression())
    model3.fit(train_x, train_y)
    pre_y = model3.predict(test_x)
    print("{}次多项式回归平均绝对误差：".format(m), mean_absolute_error(test_y, pre_y.flatten()))
    print("{}次多项式回归均方误差：".format(m), mean_squared_error(test_y, pre_y.flatten()),end="\n")
#得出结论，在本案例中，多项式回归还是优于线性回归

#选取次数为多少的回归模型 绘制MSE评价指标的图形
mse = [] #存储各多项式MSE值
m = 1 #初识m值
m_max = 10 #设定最高次数
while m<= m_max:
    model4 = make_pipeline(PolynomialFeatures(m, include_bias=False), LinearRegression())
    model4.fit(train_x, train_y)
    pre_y = model4.predict(test_x)
    mse.append(mean_squared_error(test_y, pre_y.flatten()))
    m += 1

print("MSE计算结果：",mse)
plt.plot([i for i in range(1,m_max+1)], mse, 'r')
plt.scatter([i for i in range(1, m_max+1)], mse)
plt.title("MSE of a degree of polynomial regression")
plt.xlabel("m")
plt.ylabel("MSE")
plt.show()

