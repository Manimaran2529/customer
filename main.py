import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
import seaborn as sns

mall=pd.read_csv("Mall_Customers.csv")



mall["Gender"]=mall["Gender"].map({"Male":1,"Female":0})

mall=mall.drop(columns=["CustomerID"])
#print(mall.head(10))


 

y=["Annual Income (k$)","Spending Score (1-100)"]

pt=PowerTransformer(method="yeo-johnson")
mall[y]=pt.fit_transform(mall[y])

x=mall[["Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]






k=KMeans(
    n_clusters=4,random_state=42
)

mall["cluster"]=k.fit_predict(x)

sns.scatterplot(
    data=mall,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="cluster",
    palette="tab10"
)
plt.title("Customer Clusters (K-Means)")
plt.show()
