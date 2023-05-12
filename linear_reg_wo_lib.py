import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

data=pd.read_csv("https://raw.githubusercontent.com/ojal21/ML_Dataset/main/CCPP.csv")

data.isna().sum()

#correlation 
import seaborn as sns
corr = data.corr()
sns.heatmap(data=corr, annot=True)


corr['PE'].abs().sort_values(ascending = False)

#scaling
AT=sklearn.preprocessing.scale(data.AT)
AT=pd.DataFrame(AT,columns=["AT"])

V=sklearn.preprocessing.scale(data.V)
V=pd.DataFrame(V,columns=["V"])

PE=sklearn.preprocessing.scale(data.PE)
PE=pd.DataFrame(PE,columns=["PE"])

x=pd.concat([AT,V],axis=1)
one_val=pd.DataFrame(1 for i in range(len(x)))
y=PE

x.insert(loc=0,column=0,value=one_val)

#spliting into 80-20 split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

alpha=0.0001
#theta=0
n=len(x_train)

#from scipy.sparse.construct import random
import random
cost=[]
theta=[]
#error=[]
diff=1e10
cost.append(diff)
for i in range(3):
  theta.append(random.random())

theta=np.array(theta)
theta[0]=0

x_train=np.array(x_train)
y_train=np.array(y_train)

y_train=np.reshape(y_train,len(y_train))
MSE_train=[]

#gradient descent

for i in range(100000):
  pred=np.dot(x_train,theta)
  error=pred-y_train
  #print(error.shape)
  cost_val=1/2*1/(len(y_train))*np.sum(error**2)
  cost.append(cost_val)
  MSE_train.append(2*cost_val)

  grad=1/len(y_train) * np.dot(x_train.T,error)
  theta=theta-(alpha * grad)


cost.pop(0)
#100000

import matplotlib.pyplot as plt
plt.title('Cost Function J', size = 30)
plt.xlabel('No. of iterations', size=20)
plt.ylabel('Cost', size=20)
plt.plot(cost)
plt.show()

y_test=np.array(y_test)
y_test=np.reshape(y_test,len(y_test))

#error gd
pred_test=theta[0]+theta[1]*x_test[:,1]+theta[2]*x_test[:,2]
pred_test.shape


#y_test.shape
error_test=pred_test-y_test

r2 = 1 - (sum(error_test**2)) / (sum((y_test - y_test.mean())**2))
print('R square from gradient descent: {}'.format(round(r2,4)))

mse_gd=(1/len(y_test))*np.sum(error_test**2)
mse_gd
