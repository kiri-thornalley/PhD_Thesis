import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#import datasets only required because I'm using the inbuilt iris dataset here - otherwise unecessary#
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(context='notebook', style='white', palette='viridis')

#define where dataset is located#
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

#Print the data set - lets us check we've imported the correct thing#
#print (iris)#

#Run PCA#
pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)
pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])

#Create the Scree Plot and calculate amount of variance explained by each principal component#
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_)

plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.xticks(PC_values)
plt.ylabel('Variance Explained')
plt.ylim(0,1)
plt.show()
#Automatically save the plot this generates - matplotlib currently running a non-GUI backend so cannot visualise the plot#
matplotlib.pyplot.savefig('screeplot.png')

#How much variance is explained by X number of principal components?#
print(pca.explained_variance_ratio_)
#the result of this is printed in the terminal - from this, we know that two PC are more than adequate to represent this data set.#

#Plot PCA - show first two components#
plt.figure(figsize=(8,6))
Xt = pipe.fit_transform(X)
plot = plt.scatter(Xt[:,0], Xt[:,1], c=y)
plt.legend(handles=plot.legend_elements()[0], labels=list(iris['target_names']))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Principal Component Analysis")
plt.show()
matplotlib.pyplot.savefig('PCA.png')