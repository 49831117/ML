# 複迴歸分析
# https://ithelp.ithome.com.tw/articles/10227291

import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([ [10, 80], [8, 0], [8, 200], [5, 200], [7, 300],
[8, 230], [7, 40], [9, 0], [6, 330], [9, 180] ])
y = np.array([469, 366, 371, 208, 246, 297, 363, 436,
198, 364])
reg= LinearRegression()
reg.fit(X, y) # 訓練
print("相關係數", reg.coef_) 
print("截距", reg.intercept_ )
predicted = np.array([ [10, 110] ]) # array 中放入預測的值，可放若干個
predicted_sales = reg.predict(predicted) # 預測
print("銷售額預測值（萬元）","%d" % predicted_sales)

# 評估 : model.score(data_X, data_y) 它可以對 Model 用 R^2 的方式進行評估，輸出準確率
# 準確率 越接近1 越準
# R^2 : https://en.wikipedia.org/wiki/Coefficient_of_determination
