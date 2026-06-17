import pandas as pd
import matplotlib.pyplot as plt
import folium

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("Dataset.csv")

print("Dataset Shape:", df.shape)

# ==============================
# 2. Data Cleaning
# ==============================
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing values
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Remove duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# ==============================
# 3. Data Overview
# ==============================
print("\nData Types:")
print(df.dtypes)

print("\nDataset Info:")
df.info()

# ==============================
# 4. Target Variable Analysis
# ==============================
print("\nAggregate Rating Distribution (Count):")
print(df["Aggregate rating"].value_counts())

print("\nAggregate Rating Distribution (%):")
print(df["Aggregate rating"].value_counts(normalize=True) * 100)

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

# ==============================
# 5. Visualization - Histogram
# ==============================
plt.figure(figsize=(8, 5))
plt.hist(df["Aggregate rating"], bins=10)

plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")
plt.title("Distribution of Aggregate Rating")
plt.grid(True)

# ==============================
# 6. Statistical Analysis
# ==============================
print("\nStatistical Measures:")
print(df.describe())

print("\nAggregate Rating Statistics:")
print("Mean:", df["Aggregate rating"].mean())
print("Median:", df["Aggregate rating"].median())
print("Min:", df["Aggregate rating"].min())
print("Max:", df["Aggregate rating"].max())
print("Std:", df["Aggregate rating"].std())
print("Count:", df["Aggregate rating"].count())

# ==============================
# 7. Categorical Analysis
# ==============================
print("\nCountry Distribution:")
print(df["Country Code"].value_counts())

print("\nCity Distribution:")
print(df["City"].value_counts())

print("\nTop 10 Cuisines:")
print(df["Cuisines"].value_counts().head(10))

print("\nTop 10 Cities:")
print(df["City"].value_counts().head(10))

# ==============================
# 8. Correlation Analysis
# ==============================
print("\nCorrelation (Location vs Rating):")
print(df[["Latitude", "Longitude", "Aggregate rating"]].corr())

# ==============================
# 9. Geospatial Analysis (Map)
# ==============================
map_obj = folium.Map(
    location=[df["Latitude"].mean(), df["Longitude"].mean()],
    zoom_start=5
)

for i in range(len(df)):
    folium.Marker(
        location=[df.iloc[i]["Latitude"], df.iloc[i]["Longitude"]],
        popup=df.iloc[i]["Restaurant Name"]
    ).add_to(map_obj)

map_obj.save("restaurant_map.html")
print("\nMap saved as restaurant_map.html")

# ==============================
# 10. Table Booking Analysis
# ==============================
print("\nRestaurants by City:")
print(df["City"].value_counts().head(10))

print("\nRestaurants by Country:")
print(df["Country Code"].value_counts())

table_booking_percent = df["Has Table booking"].value_counts(normalize=True) * 100
print("\nTable Booking Percentage:")
print(table_booking_percent)

print("\nAverage Rating based on Table Booking:")
print(df.groupby("Has Table booking")["Aggregate rating"].mean())

# ==============================
# 11. Online Delivery Analysis
# ==============================
online_delivery_percent = df["Has Online delivery"].value_counts(normalize=True) * 100
print("\nOnline Delivery Percentage:")
print(online_delivery_percent)

print("\nOnline Delivery by Price Range:")
print(df.groupby("Price range")["Has Online delivery"].value_counts())

print("\nOnline Delivery % by Price Range:")
print(df.groupby("Price range")["Has Online delivery"].value_counts(normalize=True) * 100)

# Bar chart
df.groupby("Price range")["Has Online delivery"] \
  .value_counts(normalize=True) \
  .unstack() \
  .plot(kind="bar")

plt.xlabel("Price Range")
plt.ylabel("Percentage")
plt.title("Online Delivery across Price Ranges")

# ==============================
# 12. Price Range Analysis
# ==============================
print("\nPrice Range Distribution:")
print(df["Price range"].value_counts())

print("\nPrice Range Percentage:")
print(df["Price range"].value_counts(normalize=True) * 100)

df["Price range"].value_counts().plot(kind="bar")

plt.xlabel("Price Range")
plt.ylabel("Number of Restaurants")
plt.title("Most Common Price Range")

# Average rating by price range
avg_rating = df.groupby("Price range")["Aggregate rating"].mean()

print("\nAverage Rating by Price Range:")
print(avg_rating)

max_range = avg_rating.idxmax()
print("\nPrice range with highest average rating:", max_range)

color = df[df["Price range"] == max_range]["Rating color"].mode()[0]
print("Color representing highest rating:", color)

# ==============================
# 13. Feature Engineering
# ==============================
df["Name Length"] = df["Restaurant Name"].apply(len)
df["Address Length"] = df["Address"].apply(len)

print("\nNew Features (Length):")
print(df[["Restaurant Name", "Name Length", "Address", "Address Length"]].head())

# Encoding
df["Table Booking (Encoded)"] = df["Has Table booking"].map({"Yes": 1, "No": 0})
df["Online Delivery (Encoded)"] = df["Has Online delivery"].map({"Yes": 1, "No": 0})

print("\nEncoded Features:")
print(df[[
    "Has Table booking", "Table Booking (Encoded)",
    "Has Online delivery", "Online Delivery (Encoded)"
]].head())

# ==============================
# 14. Show Plots
# ==============================
plt.show()