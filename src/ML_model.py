import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression

#Sets a random seed for reproducibility
np.random.seed(1234)

#Gets the new data frame with the engineered features
df = pd.read_csv("data/MLReadyDataset.csv")

df['Date'] = pd.to_datetime(df['Date'])

#Drop rows with missing demand
df = df.dropna(subset=['Demand'])

#Drops unstable features
df = df.drop(columns=['PriceToCompetitorPriceRatio',
                      'PromotionToUnitsSold',
                      'UnitsSoldToUnitsOrderedRatio',
                      'DiscountToUnitsSold',
                      'UnitsSoldToInventoryLevel',
                      'Product ID'])

df = df.sort_values("Date").reset_index(drop=True)

#Time-based Split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

def add_features(df):
    #Gets time features for Demand Forecasting
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

    #Lag Features (Critical for demand forecasting)
    df['lag_1'] = df['Demand'].shift(1)
    df['lag_7'] = df['Demand'].shift(7)
    df['lag_30'] = df['Demand'].shift(30)

    #Moving Averages 
    df['ma_7'] = df['Demand'].rolling(7).mean()
    df['ma_30'] = df['Demand'].rolling(30).mean()

    return df

df = add_features(df)

# Train/Test Split Before Encoding
train_df = add_features(train_df)
test_df = add_features(test_df)

# Fill missing lag values in test set safely
test_df[['lag_1', 'lag_7', 'lag_30', 'ma_7', 'ma_30']] = \
    test_df[['lag_1', 'lag_7', 'lag_30', 'ma_7', 'ma_30']].fillna(method='ffill').fillna(0)

# Drop rows in training where lag features are NaN
train_df = train_df.dropna()


#Define Features 
categorical_cols = ['Store ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']

numeric_cols = [col for col in df.columns
                if col not in categorical_cols + ['Demand', 'Date']]


#Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)


#Linear Regression Pipeline
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

X_train = train_df.drop(columns=['Demand'])
y_train = train_df['Demand']
X_test = test_df.drop(columns=['Demand'])
y_test = test_df['Demand']


#Fit model
model.fit(X_train, y_train)

#Evaluate model
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"Linear Regression RMSE: {rmse:.2f}")

#Cross-validation scoring
tscv = TimeSeriesSplit(n_splits=5)
cv_rmses = []

for train_idx, val_idx in tscv.split(df):
    fold_train = df.iloc[train_idx].copy()
    fold_val = df.iloc[val_idx].copy()

    fold_train = add_features(fold_train).dropna()
    fold_val = add_features(fold_val)

    fold_val[['lag_1', 'lag_7', 'lag_30', 'ma_7', 'ma_30']] = \
        fold_val[['lag_1', 'lag_7', 'lag_30', 'ma_7', 'ma_30']].fillna(method='ffill').fillna(0)

    X_fold_train = fold_train.drop(columns=['Demand'])
    y_fold_train = fold_train['Demand']
    X_fold_val = fold_val.drop(columns=['Demand'])
    y_fold_val = fold_val['Demand']

    model.fit(X_fold_train, y_fold_train)
    preds = model.predict(X_fold_val)
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, preds))
    cv_rmses.append(fold_rmse)

print(f"Cross-validated RMSE (mean over 5 folds): {np.mean(cv_rmses):.2f}")

#Save the model
joblib.dump(model, "demand_forecasting_model.pkl")



