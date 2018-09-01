import pandas as pd
from sklearn import linear_model

data=pd.read_csv("newdata.csv")
crop=input("Enter crop name: ")
x=data[data['Crop']==crop]['Area']
y=data[data['Crop']==crop]['Production']
regr=linear_model.LinearRegression()
regr.fit(x.values.reshape(-1, 1),y.values.reshape(-1,1))
coeff=regr.coef_
ar=float(input("\nEnter the value of area: "))
#prod=coeff*ar
#print("Production will be approx: ",prod)
prod2=regr.predict(ar)
print("Prediction is: ",prod2)
