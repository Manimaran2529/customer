import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from matplotlib import pyplot as plt
import seaborn as sns

mall=pd.read_csv("Mall_Customers.csv")

mall.columns=(mall.columns.str.strip())

mall=mall.drop(columns="CustomerID")
x=mall[["Annual Income (k$)","Spending Score (1-100)"]]

mall["Gender"]=mall["Gender"].map({"Male":0,"Female":1})
#print(mall["Gender"])


#plt.subplot(1,2,1)
#sns.histplot(mall["Spending Score (1-100)"])

#plt.subplot(1,2,2)
#sns.histplot(mall["Annual Income (k$)"])
#plt.show()

pt=PowerTransformer(method="yeo-johnson")


power=pt.fit_transform(x)