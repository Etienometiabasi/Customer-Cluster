# Customer-Cluster
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

# Step 1: Import dataset from kaggle
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/customer-segmentation-dataset")
print("Path to dataset files:", path)


path_to_dataset = path + "/Online Retail.xlsx"
df = pd.read_excel(path_to_dataset)
df.head()

# Step 2: Preprocess Data
# Group data by 'Country' and aggregate quantities purchased
country_quantity = df.groupby('Country').agg({'Quantity': 'sum'}).reset_index()

# One-Hot Encode 'Country'
encoder = OneHotEncoder(sparse_output=False)
country_encoded = encoder.fit_transform(country_quantity[['Country']])

# Combine encoded 'Country' with 'QuantityPurchased'
features = pd.concat([pd.DataFrame(country_encoded, columns=encoder.categories_[0]), 
                      country_quantity[['Quantity']].reset_index(drop=True)], axis=1)

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Change n_clusters as needed
country_quantity['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 4: Visualize Clusters
plt.figure(figsize=(12, 10))
sns.scatterplot(x='Quantity', y='Country', hue='Cluster', data=country_quantity, palette='viridis')
plt.title("Clusters Based on Country and Quantity Purchased")
plt.xlabel("Quantity Purchased")
plt.ylabel("Country")
plt.legend(title="Cluster")
plt.show()
