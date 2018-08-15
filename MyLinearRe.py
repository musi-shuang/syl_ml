import numpy as np
from matplotlib import pyplot as plt

x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

def f(x, w0, w1):
    y = w0+w1*x
    return y

#平方损失函数
def square_loss(x,y,w0,w1):
    loss = sum(np.square(y - (w0 + w1*x)))
    return loss

def w_calculator(x, y):
    n = len(x)
    w1 = (n*sum(x*y) - sum(x)*sum(y))/(n*sum(x*x) - sum(x)*sum(x))
    w0 = (sum(x*x)*sum(y) - sum(x)*sum(x*y))/(n*sum(x*x)-sum(x)*sum(x))
    return w0, w1


w0 = w_calculator(x, y)[0]
w1 = w_calculator(x, y)[1]

square_loss(x, y, w0, w1)

# 绘制直线自变量 [start stop]为自变量起始结束坐标点，第三个是要生成的样本数
x_temp = np.linspace(50,120,2)
plt.scatter(x,y)
plt.xlabel("Area")
plt.ylabel("Price")

#设置线条
# plt.plot(x_temp, x_temp*w1 + w0, 'r')
plt.plot(x,y, 'bo') #blue circle markers
plt.show()
# f(150, w0, w1)