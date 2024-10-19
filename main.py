import pandas as pd

# Define column names since the Iris dataset doesn't come with headers
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=columns)

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Calculate and display the mean, min, and max for sepal_length, sepal_width, petal_length, petal_width
print("\nMean values:")
print(df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].mean())

print("\nMinimum values:")
print(df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].min())

print("\nMaximum values:")
print(df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].max())

# Find and print the count of each species
species_count = df["species"].value_counts()
print("\nCount of rows for each species:")
print(species_count)
