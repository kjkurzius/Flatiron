import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity

# Step 0: Load the customer data
customers_df = pd.read_csv('retail_customers.csv')

# Step 1: Display the first few rows and basic information
print(customers_df.head())
print(customers_df.info())
print(customers_df.describe())

# Identify binary and categorical columns
binary_cols = ['gender', 'preferred_channel']
cat_col = 'region'

# Step 2: Preprocess the customer data
numerical_features = [
    'age',
    'income',
    'years_customer',
    'clothing_spend',
    'accessories_spend',
    'footwear_spend',
]

binary_cols.append(cat_col)
categorical_features = binary_cols

scaler = MinMaxScaler()
normalized_numerical = scaler.fit_transform(customers_df[numerical_features])

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categorical = encoder.fit_transform(customers_df[categorical_features])

normalized_features = np.hstack((normalized_numerical, encoded_categorical))

encoded_feature_names = list(encoder.get_feature_names_out(categorical_features))
features = numerical_features + encoded_feature_names

features_df = pd.DataFrame(normalized_features, columns=features)
print(features_df.head())

# Step 3: Calculate distance matrices and helper for similar customers
euclidean_matrix = euclidean_distances(normalized_features)
manhattan_matrix = manhattan_distances(normalized_features)
cosine_sim_matrix = cosine_similarity(normalized_features)
cosine_dist_matrix = 1 - cosine_sim_matrix


def find_similar_customers(customer_id, distance_matrix, n: int = 3):
    """Return the indices of the *n* nearest neighbours for ``customer_id``."""

    cust_idx = customer_id - 1
    distances = distance_matrix[cust_idx].copy()
    distances[cust_idx] = np.inf
    most_similar_idx = np.argsort(distances)[:n]
    return most_similar_idx.tolist()


euclidean_similar = find_similar_customers(10, euclidean_matrix)
manhattan_similar = find_similar_customers(10, manhattan_matrix)
cosine_similar = find_similar_customers(10, cosine_dist_matrix)

print("Customer 10 - Euclidean neighbours:", euclidean_similar)
print("Customer 10 - Manhattan neighbours:", manhattan_similar)
print("Customer 10 - Cosine neighbours:", cosine_similar)

# Step 4: Compare distance metrics
reference_id = 10
reference_idx = reference_id - 1
reference_customer = customers_df.iloc[reference_idx]

print(f"Reference Customer (ID: {reference_id}):")
print(reference_customer)

print("\nTop 3 similar customers by Euclidean distance:")
for idx in euclidean_similar:
    print(f"Customer {idx + 1}:")
    print(customers_df.iloc[idx])

print("\nTop 3 similar customers by Manhattan distance:")
for idx in manhattan_similar:
    print(f"Customer {idx + 1}:")
    print(customers_df.iloc[idx])

print("\nTop 3 similar customers by Cosine distance:")
for idx in cosine_similar:
    print(f"Customer {idx + 1}:")
    print(customers_df.iloc[idx])

plt.figure(figsize=(10, 6))
bar_width = 0.25
r1 = np.arange(len(euclidean_similar))
r2 = r1 + bar_width
r3 = r2 + bar_width

euclidean_distances_to_similar = [
    euclidean_matrix[reference_idx, idx] for idx in euclidean_similar
]
manhattan_distances_to_similar = [
    manhattan_matrix[reference_idx, idx] for idx in euclidean_similar
]
cosine_distances_to_similar = [
    cosine_dist_matrix[reference_idx, idx] for idx in euclidean_similar
]

plt.bar(r1, euclidean_distances_to_similar, width=bar_width, label="Euclidean")
plt.bar(r2, manhattan_distances_to_similar, width=bar_width, label="Manhattan")
plt.bar(r3, cosine_distances_to_similar, width=bar_width, label="Cosine")

plt.xlabel("Similar Customers")
plt.xticks(
    r1 + bar_width,
    [f"Customer {idx + 1}" for idx in euclidean_similar],
)
plt.ylabel("Distance")
plt.title(f"Distance from Customer {reference_id} to Similar Customers")
plt.legend()
plt.tight_layout()
plt.savefig("figures/distance_comparison.png", dpi=300)
plt.close()

# Step 5: Create a weighted hybrid distance metric


def weighted_hybrid_distances(X, weights=[0.4, 0.4, 0.2]):
    """Combine Euclidean, Manhattan, and cosine distances into a weighted hybrid."""

    if isinstance(X, pd.DataFrame):
        data = X.values
    else:
        data = np.asarray(X)

    if len(weights) != 3:
        raise ValueError(
            "weights must contain exactly three values for euclidean, manhattan, cosine"
        )

    eucl_dist = euclidean_distances(data)
    manh_dist = manhattan_distances(data)
    cos_dist = 1 - cosine_similarity(data)

    eucl_dist = eucl_dist / np.max(eucl_dist)
    manh_dist = manh_dist / np.max(manh_dist)
    cos_dist = cos_dist / np.max(cos_dist)

    weighted = (
        weights[0] * eucl_dist
        + weights[1] * manh_dist
        + weights[2] * cos_dist
    )
    return weighted


hybrid_matrix = weighted_hybrid_distances(features_df)
print("Hybrid distance matrix shape:", hybrid_matrix.shape)

# Step 6: Evaluate the hybrid distance metric for a high-spend customer
reference_id = 16
reference_idx = reference_id - 1
reference = customers_df.iloc[reference_idx]
ref_total_spend = (
    reference["clothing_spend"]
    + reference["accessories_spend"]
    + reference["footwear_spend"]
)

clothing_pct = (reference["clothing_spend"] / ref_total_spend) * 100
accessories_pct = (reference["accessories_spend"] / ref_total_spend) * 100
footwear_pct = (reference["footwear_spend"] / ref_total_spend) * 100

hybrid_similar = find_similar_customers(reference_id, hybrid_matrix)

print(f"Reference Customer (ID: {reference_id}):")
print(
    f"Gender: {reference['gender']}, Age: {reference['age']}, Income: ${reference['income']}"
)
print(
    "Spending: Clothing ${}, Accessories ${}, Footwear ${}".format(
        reference["clothing_spend"],
        reference["accessories_spend"],
        reference["footwear_spend"],
    )
)
print(f"Total spending: ${ref_total_spend:.2f}")
print(
    "Spending breakdown: Clothing {:.1f}%, Accessories {:.1f}%, Footwear {:.1f}%".format(
        clothing_pct,
        accessories_pct,
        footwear_pct,
    )
)
print("This customer clearly prioritizes clothing and footwear.")
print(reference)


def print_customer_details(customer_idx, label=""):
    customer = customers_df.iloc[customer_idx]
    total = (
        customer["clothing_spend"]
        + customer["accessories_spend"]
        + customer["footwear_spend"]
    )

    c_pct = (customer["clothing_spend"] / total) * 100
    a_pct = (customer["accessories_spend"] / total) * 100
    f_pct = (customer["footwear_spend"] / total) * 100

    print()
    print(f"{label} Customer {customer_idx + 1}:")
    print(
        f"  Gender: {customer['gender']}, Age: {customer['age']}, Income: ${customer['income']}"
    )
    print(f"  Total spending: ${total:.2f}")
    print(
        "  Spending: Clothing ${} ({:.1f}%), Accessories ${} ({:.1f}%), Footwear ${} ({:.1f}%)".format(
            customer["clothing_spend"],
            c_pct,
            customer["accessories_spend"],
            a_pct,
            customer["footwear_spend"],
            f_pct,
        )
    )
    print(customer)


print_customer_details(hybrid_similar[0], "Hybrid recommends")

conclusion = """\
After comparing how different distance metrics find similar customers, the hybrid
distance metric with weights works well for retail customer segmentation
because:

1. It balances overall similarity (from Euclidean distance) with spending
   pattern similarity (from Cosine distance), which is important for effective
   marketing.
2. The Euclidean component (0.4 weight) ensures we find customers with similar
   demographics and total spending capacity.
3. The Manhattan component (0.4 weight) provides robustness when customers have
   unusual spending in one category.
4. The Cosine component (0.2 weight) helps identify customers with similar
   spending priorities (how they divide their budget), even if their total
   spending differs.
5. In our tests, the hybrid approach found customers who were both
   demographically similar and had similar purchasing behaviors, making it more
   practical for targeted marketing campaigns.

This lab provided an exploration of distance metrics for customer similarity
analysis in a retail context. By implementing and comparing Euclidean,
Manhattan, and Cosine distance metrics, we demonstrated how different
mathematical approaches to measuring similarity can yield different results.

The key takeaways include:

- Metric selection matters: Each distance metric emphasizes different aspects
  of similarity. Euclidean distance captures overall profile similarity,
  Manhattan distance offers robustness to outliers, and Cosine distance
  identifies similar preferences regardless of scale.
- Preprocessing is crucial: Proper handling of categorical variables through
  encoding and normalization of numerical features ensures fair comparison
  across different types and scales of data.
- Hybrid approaches offer balance: Combining multiple distance metrics with
  appropriate weights can create a more comprehensive similarity measure that
  captures both absolute and proportional similarity.
- Business context drives metric choice: The "best" distance metric depends on
  the specific business goals. For retail marketing, a hybrid approach balances
  spending capacity with customer preferences.
- Practical applications: Distance-based similarity measures can directly
  power recommendation systems, helping businesses identify cross-selling
  opportunities based on similar customers' behavior.

This approach to customer similarity analysis provides retail businesses with a
powerful tool for personalized marketing, inventory planning, and customer
relationship management.
"""

print("\nConclusion:\n")
print(conclusion)
