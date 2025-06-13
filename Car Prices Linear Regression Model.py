import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split    
df=pd.read_csv('car_price_dataset.csv')
x=df[['Mileage_km']]
y=df['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x)
plt.title('Car Prices Linear Regrssion Model')
plt.xlabel('Mileage')
plt.ylabel('km')
plt.scatter(x,y,color='blue')
plt.plot(x,y_pred)
plt.show()
