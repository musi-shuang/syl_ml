import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("D:\py\challenge-1-beijing.csv")
print(df.head())
# features = df[['公交','写字楼','医院','商场','地铁','学校','建造时间','楼层','面积']]
# 或去除数据
features = df[df.columns.drop(['小区名字','房型','每平米价格'])]
target = df['每平米价格']
split_num = int(len(df)*0.7)
train_x = features[:split_num]
train_y = target[:split_num]
test_x = features[split_num:]
test_y = target[split_num:]

model = LinearRegression()
model.fit(train_x, train_y)

# 评价指标选择平均绝对百分比误差MAPE
def mape(y_true, y_pred):
    n = len(y_true)
    mape = np.sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape

y_pred = model.predict(test_x)
print(mape(test_y, y_pred))
# 45.50618854676238