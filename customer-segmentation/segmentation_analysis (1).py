"""
Customer Segmentation Project — Stationery Sales Dataset
==========================================================
NOTE ON DATA ADAPTATION:
The uploaded dataset (stationery_sales_dataset.csv) is a B2B sales transaction
log (Brand -> Retailer), NOT an individual end-consumer dataset — there is no
customer ID, age, gender or income column.

To still deliver a genuine "customer segmentation" project on this data, each
(Retailer, Region) combination is treated as a "customer" — i.e. we segment
the brand's retail-channel customers (5 retailers x 5 regions = 25 buying
units) based on their purchasing BEHAVIOR: how much they buy, how often, at
what price point, how discount-sensitive they are, and how diverse their
category mix is. This is a standard and realistic approach for B2B / channel
sales analytics (segmenting "accounts" instead of individual shoppers).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 130

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv("stationery_sales_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])

# ------------------------------------------------------------------
# 2. BUILD "CUSTOMER" (Retailer x Region) FEATURE TABLE
# ------------------------------------------------------------------
grp = df.groupby(['Retailer', 'Region'])

features = grp.agg(
    Total_Revenue=('Revenue (INR)', 'sum'),
    Total_Units=('Units Sold', 'sum'),
    Num_Transactions=('Sale ID', 'count'),
    Avg_Unit_Price=('Unit Price (INR)', 'mean'),
    Avg_Discount=('Discount (%)', 'mean'),
    Category_Diversity=('Category', pd.Series.nunique),
    Brand_Diversity=('Brand', pd.Series.nunique),
).reset_index()

features['Avg_Order_Value'] = features['Total_Revenue'] / features['Num_Transactions']

# Preferred (most revenue-generating) category per customer, for profiling
top_cat = (df.groupby(['Retailer', 'Region', 'Category'])['Revenue (INR)']
             .sum().reset_index()
             .sort_values('Revenue (INR)', ascending=False)
             .drop_duplicates(['Retailer', 'Region'])
             .rename(columns={'Category': 'Top_Category'})
             [['Retailer', 'Region', 'Top_Category']])

features = features.merge(top_cat, on=['Retailer', 'Region'])
features['Customer'] = features['Retailer'] + " - " + features['Region']

print("Feature table (first 10 rows):")
print(features.head(10).to_string(index=False))
features.to_csv("customer_features.csv", index=False)

# ------------------------------------------------------------------
# 3. SCALE FEATURES FOR CLUSTERING
# ------------------------------------------------------------------
cluster_cols = ['Total_Revenue', 'Total_Units', 'Num_Transactions',
                 'Avg_Unit_Price', 'Avg_Discount', 'Category_Diversity',
                 'Avg_Order_Value']

X = features[cluster_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------
# 4. FIND OPTIMAL K (Elbow + Silhouette)
# ------------------------------------------------------------------
inertias, sil_scores = [], []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(K_range), inertias, marker='o', color='#4C72B0')
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")

axes[1].plot(list(K_range), sil_scores, marker='o', color='#DD8452')
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("chart_elbow_silhouette.png", bbox_inches='tight')
plt.close()

best_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"\nBest k by silhouette score: {best_k}")

# ------------------------------------------------------------------
# 5. FINAL CLUSTERING (k=3 chosen for interpretability, unless silhouette
#    strongly prefers otherwise)
# ------------------------------------------------------------------
FINAL_K = 3
kmeans = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
features['Segment'] = kmeans.fit_predict(X_scaled)

seg_names = (features.groupby('Segment')['Total_Revenue'].mean()
             .sort_values(ascending=False).index)
rank_map = {seg: name for seg, name in zip(seg_names, ["High-Value", "Mid-Value", "Low-Value"])}
features['Segment_Label'] = features['Segment'].map(rank_map)

print("\nFinal segment counts:")
print(features['Segment_Label'].value_counts())

# ------------------------------------------------------------------
# 6. PCA VISUALIZATION (2D projection of clusters)
# ------------------------------------------------------------------
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
features['PC1'], features['PC2'] = pcs[:, 0], pcs[:, 1]

plt.figure(figsize=(7, 5.5))
palette = {"High-Value": "#2E7D32", "Mid-Value": "#F9A825", "Low-Value": "#C62828"}
sns.scatterplot(data=features, x='PC1', y='PC2', hue='Segment_Label',
                 palette=palette, s=140, edgecolor='black', linewidth=0.6)
for _, row in features.iterrows():
    plt.text(row['PC1']+0.05, row['PC2']+0.05, row['Retailer'][:4]+"-"+row['Region'][:1],
              fontsize=7, alpha=0.7)
plt.title("Customer Segments (PCA Projection)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(title="Segment")
plt.tight_layout()
plt.savefig("chart_pca_segments.png", bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------
# 7. SEGMENT PROFILE (bar charts of key metrics per segment)
# ------------------------------------------------------------------
profile = features.groupby('Segment_Label')[cluster_cols].mean().reindex(
    ["High-Value", "Mid-Value", "Low-Value"])
profile.to_csv("segment_profile.csv")

fig, axes = plt.subplots(2, 4, figsize=(15, 7))
axes = axes.flatten()
colors = [palette[i] for i in profile.index]
for i, col in enumerate(cluster_cols):
    axes[i].bar(profile.index, profile[col], color=colors)
    axes[i].set_title(col.replace('_', ' '))
    axes[i].tick_params(axis='x', rotation=15)
axes[-1].axis('off')
plt.suptitle("Segment Profiles — Average Feature Values", fontsize=14)
plt.tight_layout()
plt.savefig("chart_segment_profiles.png", bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------
# 8. PURCHASE PATTERN ANALYSIS
# ------------------------------------------------------------------
# 8a. Revenue by Category
cat_rev = df.groupby('Category')['Revenue (INR)'].sum().sort_values(ascending=False)
plt.figure(figsize=(6.5, 4.5))
sns.barplot(x=cat_rev.values, y=cat_rev.index, palette="Blues_r")
plt.title("Total Revenue by Product Category")
plt.xlabel("Revenue (INR)")
plt.tight_layout()
plt.savefig("chart_revenue_by_category.png", bbox_inches='tight')
plt.close()

# 8b. Monthly revenue trend
monthly = df.set_index('Date').resample('ME')['Revenue (INR)'].sum()
plt.figure(figsize=(8, 4.5))
monthly.plot(marker='o', color='#4C72B0')
plt.title("Monthly Revenue Trend (2026)")
plt.ylabel("Revenue (INR)")
plt.xlabel("Month")
plt.tight_layout()
plt.savefig("chart_monthly_trend.png", bbox_inches='tight')
plt.close()

# 8c. Segment x Category preference heatmap
seg_cat = (df.merge(features[['Retailer', 'Region', 'Segment_Label']],
                     on=['Retailer', 'Region'])
             .groupby(['Segment_Label', 'Category'])['Revenue (INR)'].sum()
             .unstack(fill_value=0))
seg_cat = seg_cat.reindex(["High-Value", "Mid-Value", "Low-Value"])
seg_cat_pct = seg_cat.div(seg_cat.sum(axis=1), axis=0) * 100

plt.figure(figsize=(7, 4))
sns.heatmap(seg_cat_pct, annot=True, fmt='.0f', cmap="YlGnBu", cbar_kws={'label': '% of segment revenue'})
plt.title("Category Preference by Segment (%)")
plt.tight_layout()
plt.savefig("chart_segment_category_heatmap.png", bbox_inches='tight')
plt.close()

# 8d. Discount sensitivity vs revenue scatter
plt.figure(figsize=(6.5, 5))
sns.scatterplot(data=features, x='Avg_Discount', y='Total_Revenue',
                 hue='Segment_Label', palette=palette, s=130, edgecolor='black')
plt.title("Discount Sensitivity vs Total Revenue")
plt.xlabel("Average Discount Given (%)")
plt.ylabel("Total Revenue (INR)")
plt.tight_layout()
plt.savefig("chart_discount_vs_revenue.png", bbox_inches='tight')
plt.close()

features.to_csv("customer_features_segmented.csv", index=False)

print("\nAll charts and CSVs generated successfully.")
print("\n--- SEGMENT PROFILE SUMMARY ---")
print(profile.round(1).to_string())
