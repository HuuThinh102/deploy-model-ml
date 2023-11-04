import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


# Read original dataset
df = pd.read_csv("data/Dry_Bean_Dataset.csv")


#drop unuse column
df = df.drop(['ShapeFactor5'], axis=1)

#drop duplicate rows
df.drop_duplicates(inplace=True)

#remove empty cells
df.dropna(inplace=True)

#replace ',' to '.'
df['Compactness'] = df['Compactness'].replace(',','.', regex=True)
df['Compactness'] = df['Compactness'].astype(float)
df['ShapeFactor3'] = df['ShapeFactor3'].replace(',','.', regex=True)
df['ShapeFactor3'] = df['ShapeFactor3'].astype(float)


#change 'Class' column from String to numerical
l1 = LabelEncoder()
df['Class'] = l1.fit_transform(df['Class'])

#split dataset
X = df.drop('Class', axis = 1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# create an instance of the random forest classifier
model = RandomForestClassifier(n_estimators=100)

# train the classifier on the training data
model.fit(X_resampled, y_resampled)

# predict on the test set
# y_pred = model.predict(X_test_scaled)

# calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")

# save the model to disk

joblib.dump(scaler, 'scale_func.sav')
joblib.dump(model, "rf_model.sav")

