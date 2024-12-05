import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pickle
import warnings
warnings.filterwarnings("ignore")

#Step 1: Collect and preprocess the data
data = pd.read_csv("TS Weather data January 2022.csv")
X = data.drop(['District', 'Mandal', 'Rain (mm)', 'Min Temp (°C)', 'Max Temp (°C)', 'Min Humidity (%)',
              'Max Humidity (%)', 'Min Wind Speed (Kmph)', 'Max Wind Speed (Kmph)', 'Temperature'], axis=1)
data = data.dropna()

# step 2: split the date into yr,mnth,day
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data.drop(['Date'], axis=1, inplace=True)
train_data, test_data = train_test_split(data, test_size=0.2)

#Step 3: Create the model
model_pd = SVR(C=1.0, epsilon=0.2)

#Step 4: Train the model
model_pd.fit(train_data[['year', 'month', 'day']], train_data[['Temperature']])

#Step 5: Evaluate the model
predictions = model_pd.predict(test_data[['year', 'month', 'day']])
mse = mean_squared_error(test_data[['Temperature']], predictions)
# print("Mean Squared Error: ", mse)

#Step 6: Make predictions
# new_data = pd.DataFrame({'year': [2022], 'month': [7], 'day': [21]})
# predictions = model_pd.predict(X=new_data)
# print("Predicted temperature: ", predictions)

pickle.dump(model_pd,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))