import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.metrics  import mean_squared_error

# 预测比特币的走势,只使用比特币市场价格、比特币总量和比特币交易费用三个字段
df = pd.read_csv("D:\py\challenge-2-bitcoin.csv")
# print(df.head())
data = df[['btc_market_price','btc_total_bitcoins', 'btc_transaction_fees']]

fig, axes = plt.subplots(1,3,figsize=(16,5))
axes[0].plot(data['btc_market_price'],'green')
axes[0].set_xlabel('time')
axes[0].set_ylabel('btn_market_price')

axes[1].plot(data['btc_total_bitcoins'],'blue')
axes[1].set_xlabel('time')
axes[1].set_ylabel('btc_total_bitcoins')

axes[2].plot(data['btc_transaction_fees'],'brown')
axes[2].set_xlabel('time')
axes[2].set_ylabel('btc_transaction_fees')

plt.show()

# 划分数据集
def split_dataset():
    train_data = data[:int(len(data)*0.7)]
    test_data = data[int(len(data)*0.7):]

    train_x = train_data[['btc_total_bitcoins','btc_transaction_fees']]
    train_y = train_data[['btc_market_price']]

    test_x = test_data[['btc_total_bitcoins','btc_transaction_fees']]
    test_y = test_data[['btc_market_price']]

    return train_x, train_y, test_x, test_y
train_x, train_y, test_x, test_y = split_dataset()
print(len(train_x), len(train_y), len(test_x), len(test_y),
      train_x.shape, train_y.shape, test_x.shape, test_y.shape)
# 构建3次多项式回归预测模型
def poly3():
    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    poly_train_x = poly_features.fit_transform(train_x)
    poly_test_x = poly_features.fit_transform(test_x)

    model = LinearRegression()
    model.fit(poly_train_x, train_y)
    pre_y = model.predict(poly_test_x)

    mae = mean_absolute_error(test_y, pre_y.flatten())
    return mae

print(poly3())

# N次多项式回归预测模型
def poly_plot(N):
    m = 1
    mse = []
    while m <= N:
        model2 = make_pipeline(PolynomialFeatures(m, include_bias=False),LinearRegression())
        model2.fit(train_x, train_y)
        pre_y = model2.predict(test_x)
        mse.append(mean_squared_error(test_y, pre_y.flatten()))
        m += 1
    return mse
print(poly_plot(10)[:10:3])
# 绘制mse图形
mse = poly_plot(10)
plt.plot([i for i in range(1,11)], mse, 'r')
plt.scatter([i for i in range(1,11)],mse)

plt.title("MSE")
plt.xlabel("N")
plt.ylabel("MSE")
plt.show()










