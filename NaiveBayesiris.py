import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# ----------------------------------
# Load Dataset
# ----------------------------------

df = pd.read_csv("Iris.csv")

print("\nIRIS DATASET\n")
print(df.head())

# ----------------------------------
# Check Missing Values
# ----------------------------------

print("\nMissing Values\n")
print(df.isnull().sum())

# ----------------------------------
# Remove Duplicate Records
# ----------------------------------

before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print("\nDuplicates Removed:", before - after)

# ----------------------------------
# Remove Id Column
# ----------------------------------

df.drop("Id", axis=1, inplace=True)

# ----------------------------------
# Encode Target Variable
# ----------------------------------

encoder = LabelEncoder()

df["Species"] = encoder.fit_transform(df["Species"])

# ----------------------------------
# Features and Target
# ----------------------------------

X = df[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]]

y = df["Species"]

# ----------------------------------
# Split Dataset
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------
# Train Naive Bayes Model
# ----------------------------------

model = GaussianNB()

model.fit(X_train, y_train)

# ----------------------------------
# Model Evaluation
# ----------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy :", round(accuracy * 100, 2), "%")

# ----------------------------------
# User Prediction
# ----------------------------------

print("\n========== Enter Flower Details ==========\n")

sepal_length = float(input("Sepal Length (cm): "))
sepal_width = float(input("Sepal Width (cm): "))
petal_length = float(input("Petal Length (cm): "))
petal_width = float(input("Petal Width (cm): "))

sample = pd.DataFrame({
    "SepalLengthCm": [sepal_length],
    "SepalWidthCm": [sepal_width],
    "PetalLengthCm": [petal_length],
    "PetalWidthCm": [petal_width]
})

prediction = model.predict(sample)

species = encoder.inverse_transform(prediction)

print("\n========== Prediction ==========\n")
print("Predicted Species :", species[0])

# ----------------------------------
# Confusion Matrix
# ----------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# ----------------------------------
# Class Distribution
# ----------------------------------

species_count = df["Species"].value_counts().sort_index()

labels = encoder.inverse_transform(species_count.index)

plt.figure(figsize=(6,5))

plt.bar(labels, species_count.values)

plt.title("Distribution of Iris Species")

plt.xlabel("Species")

plt.ylabel("Number of Flowers")

plt.grid(axis="y")

plt.show()