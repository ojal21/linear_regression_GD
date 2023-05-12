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

x_train=np.array(x_train)
y_train=np.array(y_train)

y_train=np.reshape(y_train,len(y_train))


y_test=np.array(y_test)
y_test=np.reshape(y_test,len(y_test))


from sklearn.linear_model import LinearRegression

lr=LinearRegression()

model=lr.fit(x_train,y_train)
x_test=np.array(x_test)
pred_mod=lr.predict(x_test)

#error mmodel
r2_sk = lr.score(x_test,y_test)
print('R square from sci-kit learn: {}'.format(round(r2_sk,4)))



err_mod=pred_mod-y_test
mse_mod=(1/len(y_test))*np.sum(err_mod**2)
mse_mod
