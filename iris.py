import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


iris_data = pd.read_csv("tableconvert_csv_rz1wsc.csv", encoding="utf-8")


y = iris_data.loc[:,"species"]
x = iris_data.loc[:,["sepal_length", "sepal_width", "petal_length", "petal_width"]]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,train_size=0.8,shuffle = True)


clf = SVC()
clf.fit(x_train, y_train)


y_pred = clf.predict(x_test)
print("accurate = ", accuracy_score(y_test,y_pred))