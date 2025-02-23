import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense

data_file="concrete_data.csv"

concrete_df = pd.read_csv(data_file)
test_concrete_df= pd.read_csv("test_concrete_data.csv")

predictors = concrete_df.iloc[:, :-1]  # Predictor variables (all columns except the last one)
target = concrete_df.iloc[:, -1]   # Target variable (the last column)

# initialize a sequential model
model = Sequential()

n_cols= concrete_df.shape[1]

# print(n_cols)
# print(predictors.shape[1])
# print(target.shape)
# print(target.head())

# Add layer to the neural neywork

model.add(Dense(5, activation='relu', input_shape=(8,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

# define the optimizer and error metric
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(predictors, target)

prediction= model.predict(test_concrete_df)

print(prediction)


