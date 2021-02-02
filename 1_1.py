## https://scikit-learn.org/stable/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 觀察資料
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
plt.figure()
plt.title('Pizza Price with diameter.')
plt.xlabel('diameter(inch)')
plt.ylabel('price($)')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.plot(X, y, 'k.')
# pyplot.plot 第3個參數是format string 
# https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html
plt.show()


# 建立模型與評量
reg = LinearRegression()
# X and y is the data in previous code.
reg.fit(X, y)
print(u'係數', reg.coef_)
print (u'截距', reg.intercept_)
print (u'評分函式', reg.score(X, y))
X2 = [[1], [10], [14], [25]]
y2 = reg.predict(X2)
print(y2)


# 資料預測
X2 = [[1], [10], [14], [25]]
y2 = reg.predict(X2)
print(y2)
#繪製線性迴歸圖形
plt.figure()
plt.title(u'Pizza Price with diameter.') #標題
plt.xlabel(u'diameter') #x軸座標
plt.ylabel(u'price') #y軸座標
plt.axis([0, 25, 0, 25]) #區間
plt.grid(True) #顯示網格
plt.plot(X, y, 'k.') #繪製訓練資料集散點圖
plt.plot(X2, y2, '-g.') #繪製預測資料集直線
