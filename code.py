import numpy as np #[Harris, C.R. et al, 2020]
import pandas as pd #[McKinney 2010]
import matplotlib.pyplot as plt #[Hunter, J.D, 2007]
import seaborn as sea #[Waskom, M. L., 2021]
import sklearn.model_selection as sk #[Pedregosa et al., 2011]
import sklearn.metrics as skm
import sklearn.preprocessing as skp
from sklearn import svm

train=pd.read_csv('mnist_train.csv') #this is taken from [LeCun et al., 1998a]
sam=int(0.2*60000) # to save time use 20% of the data set for training
data=train.sample(sam)

print(sea.countplot(x='label', data=data)) # to see distribtution of numbers in sample

labels=data['label'] #first row contains the labels of the data
images=data.drop('label',axis=1) # these are the pixels coresponding to the numbers
scaler = skp.StandardScaler() #scikit learn standard scaler used to fit the images pixels
images=scaler.fit_transform(images)

clf=svm.SVC(kernel='poly',degree=3)

comp=clf.fit(images,labels)

test=pd.read_csv('mnist_test.csv').sample(1000)
test_x=test.drop('label',axis=1)
test_x=scaler.fit_transform(test_x) #scale these like for the training set
test_y=test['label']

sea.countplot(x='label', data=test)

predic=comp.predict(test_x)
conf=skm.confusion_matrix(test_y.values,predic)

print(skm.classification_report(test_y,predic))
print(conf)

dat=test.drop('label',axis=1).to_numpy()
results=test_y.to_numpy()

number1=dat[0].reshape(28,28)
number2=dat[1].reshape(28,28)
number3=dat[2].reshape(28,28)
number4=dat[3].reshape(28,28)

fig,(ax1,ax2,ax3,ax4)=plt.subplots(nrows=1, ncols=4, sharey=True ,sharex=True)
#mpl.axis.XAxis.set_ticks_position(position='none')

ax1.set_title('Pred:{} Lab:{}'.format(predic[0],results[0]))
ax1.matshow(number1)

ax2.set_title('Pred:{} Lab:{}'.format(predic[1],results[1]))
ax2.matshow(number2)

ax3.set_title('Pred:{} Lab:{}'.format(predic[2],results[2]))
ax3.matshow(number3)

ax4.set_title('Pred:{} Lab:{}'.format(predic[3],results[3]))
ax4.matshow(number4)

#Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
#Waskom, M. L., (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021, https://doi.org/10.21105/joss.03021.
# Pedregosa et al., Scikit-learn: Machine Learning in Python,JMLR 12, pp. 2825-2830, 2011.
# McKinney, Data structures for statistical computing in python, Proceedings of the 9th Python in Science Conference, Volume 445, 2010.
# J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
#Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. 
