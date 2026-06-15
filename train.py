import pandas as pd
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# -----------------------------
# Load Dataset
# -----------------------------

data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target


# -----------------------------
# Feature Engineering
# -----------------------------

# Remove duplicate rows
df.drop_duplicates(inplace=True)


# Remove highly correlated columns

corr_matrix = df.drop("target", axis=1).corr()

columns_to_remove = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.90:
            colname = corr_matrix.columns[i]
            columns_to_remove.add(colname)


print("Removed Columns:")
print(columns_to_remove)


df.drop(
    columns=columns_to_remove,
    inplace=True
)


# Split X and y

X = df.drop("target", axis=1)

y = df["target"]


# Save final selected features

with open("features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)



# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# -----------------------------
# Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)



# -----------------------------
# ANN Model
# -----------------------------

model = Sequential()


model.add(
    Dense(
        16,
        activation="relu",
        input_dim=X_train.shape[1]
    )
)


model.add(
    Dense(
        8,
        activation="relu"
    )
)


model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)



model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)



model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test,y_test)
)



loss,accuracy=model.evaluate(
    X_test,
    y_test
)

print("Accuracy:",accuracy)



# Save ANN model

model.save("ann_model.keras")

print("Training completed")