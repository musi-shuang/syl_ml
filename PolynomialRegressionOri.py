from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import leastsq
# # 二次项回归
x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]
y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]
# # plt.scatter(x,y)
# # plt.show()
#
# # 实现二次拟合
# def twofunc(p,x):
#     w0,w1,w2 = p
#     f = w0 + w1*x + w2*x*x
#     return f
#
# def err_func(p,x ,y):
#     ret = twofunc(p,x) - y
#     return ret
#
# p_init = np.random.randn(3) # 生成三个随机数，初始化参数
# # 使用Scipy提供的最小二乘法函数得到最佳拟合函数
# parameters = leastsq(err_func, p_init, args = (np.array(x), np.array(y)))
# print(parameters[0])
#
# # 绘制2次多项式拟合曲线
# x_tmp = np.linspace(0,80,10000)
# plt.plot(x_tmp, twofunc(parameters[0], x_tmp),'r')
# plt.show()


#实现N次多项式拟合
def fit_func(p,x):
    f = np.poly1d(p)
    return f(x)

def err_func(p, x, y):
    ret = fit_func(p,x)-y
    return ret

def n_poly(n):
    p_init = np.random.randn(n)
    parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))
    return parameters[0]
n_poly(3)

#绘制出次数为4,5,6,7,8,9多项次的拟合图像
x_temp = np.linspace(0, 80, 10000)
# 绘制子图 两行三列 figure代表整体图形比例
fig, axes = plt.subplots(2, 3, figsize=(15,10))

axes[0,0].plot(x_temp, fit_func(n_poly(4), x_temp), 'r')
axes[0,0].scatter(x, y)
axes[0,0].set_title("m = 4")

axes[0,1].plot(x_temp, fit_func(n_poly(5), x_temp), 'r')
axes[0,1].scatter(x, y)
axes[0,1].set_title("m = 5")

axes[0,2].plot(x_temp, fit_func(n_poly(6), x_temp), 'r')
axes[0,2].scatter(x, y)
axes[0,2].set_title("m = 6")

axes[1,0].plot(x_temp, fit_func(n_poly(7), x_temp), 'r')
axes[1,0].scatter(x, y)
axes[1,0].set_title("m = 7")

axes[1,1].plot(x_temp, fit_func(n_poly(8), x_temp), 'r')
axes[1,1].scatter(x, y)
axes[1,1].set_title("m = 8")

axes[1,2].plot(x_temp, fit_func(n_poly(9), x_temp), 'r')
axes[1,2].scatter(x, y)
axes[1,2].set_title("m = 9")
print(fig) # Figure(1500x1000)
plt.show()

# 可以看出当n= 7时已经明显的过拟合

