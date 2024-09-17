import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense ,Input

train = pd.read_csv('Train.csv')
print("data Read Successfully")
print(train.head())
print(train.info())

# Fill missing values in 'bath' with the median value
train['bath'] = train['bath'].fillna(train['bath'].median())

# Fill missing values in 'balcony' with the median value
train['balcony'] = train['balcony'].fillna(train['balcony'].median())

# Fill missing values in 'location' with the mode value
train['location'] = train['location'].fillna(train['location'].mode()[0])

# Fill missing values in 'size' with the mode value
train['size'] = train['size'].fillna(train['size'].mode()[0])


train.drop('society', axis=1, inplace=True)

#missing values find
missing_values = train.isnull().sum()
print(missing_values[missing_values > 0])



categorical_cols = ['location', 'size','area_type','availability'] 
# Initialize OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform categorical columns
cat_encoded = ohe.fit_transform(train[categorical_cols])

# Convert to DataFrame
cat_encoded_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(categorical_cols))

# Drop original categorical columns and concatenate new ones
train = train.drop(categorical_cols, axis=1)
train = pd.concat([train, cat_encoded_df], axis=1)

print("Data after one-hot encoding:")
print(train.head())


def convert_sqft_to_num(x):
    try:
        # If it's a range, take the average
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        # If it's a single number, return it as a float
        return float(x)
    except:
        # Handle any other cases (e.g., non-numeric values)
        return None

# Apply the function to 'total_sqft'
train['total_sqft'] = train['total_sqft'].apply(convert_sqft_to_num)

# Drop rows with missing or invalid 'total_sqft' values
train = train.dropna(subset=['total_sqft'])

scaler = MinMaxScaler()

# Normalize features (excluding the 'price' column)
norm_trainX = scaler.fit_transform(train.drop('price', axis=1))

# Normalize the target variable 'price'
norm_trainY = scaler.fit_transform(train[['price']])

print("Normalized Training Features (X):")
print(norm_trainX[:5])

print("\nNormalized Training Target (Y):")
print(norm_trainY[:5])

#Neural Model
model = Sequential([
    Input(shape=(norm_trainX.shape[1],)), 
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(norm_trainX, norm_trainY, batch_size=32, epochs=50, validation_split=0.2)

#Loss calculate
loss = model.evaluate(norm_trainX, norm_trainY)
print(f"Model Loss: {loss}")

predictions = model.predict(norm_trainX
)

#for printing first few predictions
predictions_denorm = scaler.inverse_transform(predictions)

# Display the first few predictions
print("Predicted House Prices:")
print(predictions_denorm[:5])

model.save('house_price_model_Shayaan.h5')





