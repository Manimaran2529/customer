import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
import seaborn as sns

mall=pd.read_csv("Mall_Customers.csv")



mall["Gender"]=mall["Gender"].map({"Male":1,"Female":0})

mall=mall.drop(columns=["CustomerID"])
#print(mall.head(10))

x=mall[["Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]
 

y=mall[["Annual Income (k$)","Spending Score (1-100)"]]

pt=PowerTransformer(method="yeo-johnson")
x_train=pt.fit_transform(y)
x_test=pt.transform(y)


print(x_train)
