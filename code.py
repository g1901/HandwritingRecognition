import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import sklearn.model_selection as sk
import sklearn.metrics as skm
import sklearn.preprocessing as skp
from sklearn import svm

train=pd.read_csv('mnist_train.csv')
sam=int(0.2*60000)
data=train.sample(sam)

dat=data.drop('label',axis=1).to_numpy()
number=dat[0]
number=number.reshape(28,28)
lab=data['label'].to_numpy()
print(lab[0])
plt.matshow(number)

sea.countplot(x='label', data=data)

labels=data['label']
images=data.drop('label',axis=1)
scaler = skp.StandardScaler()
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

import  PIL as pil
own=pil.Image.open("drawing1.png") # need to do a file for this
own

own=scaler.fit_transform(np.array(own).reshape(1,-1))
pred=comp.predict(own)
print(pred)
