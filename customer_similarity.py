"""Customer similarity analysis for a retail dataset.

This module follows a step-by-step lab structure:

* load and inspect the raw ``retail_customers.csv`` file
* identify which columns are binary versus multi-category
* normalize numeric features and one-hot encode categoricals
* compute Euclidean, Manhattan, and cosine distance matrices
* build a hybrid metric that averages the individual distances
* expose helper utilities to retrieve similar customers and export reports

The implementation avoids third-party dependencies so that it runs in
restricted environments where packages such as pandas, NumPy, or
scikit-learn are unavailable.  All preprocessing and metric calculations
are implemented with the Python standard library.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CustomerRecord:
    """Container for a single customer's information."""

    customer_id: int
    features: Dict[str, object]


@dataclass
class PreprocessedData:
    """Feature engineering output used for similarity computations."""

    customer_ids: List[int]
    feature_names: List[str]
    feature_matrix: List[List[float]]


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

ID_FIELD = "customer_id"

NUMERIC_FEATURES = [
    "income",
    "age",
    "years_customer",
    "clothing_spend",
    "accessories_spend",
    "footwear_spend",
]

BINARY_FEATURES = ["gender", "preferred_channel"]

MULTI_CATEGORY_FEATURES = ["region"]

CATEGORICAL_FEATURES = BINARY_FEATURES + MULTI_CATEGORY_FEATURES

# ---------------------------------------------------------------------------
# Step 1 answers (binary and categorical columns)
# ---------------------------------------------------------------------------

binary_cols = BINARY_FEATURES.copy()
cat_col = MULTI_CATEGORY_FEATURES[0]


def load_customer_data(path: Path) -> List[CustomerRecord]:
    """Load customer data from a CSV file."""

    records: List[CustomerRecord] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return records

        for row in reader:
            normalized_row = {
                key.strip().lower(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
            }

            if ID_FIELD not in normalized_row:
                raise KeyError(f"CSV is missing required column '{ID_FIELD}'")

            customer_id = int(float(normalized_row[ID_FIELD]))

            features: Dict[str, object] = {}
            for feature in NUMERIC_FEATURES:
                try:
                    features[feature] = float(normalized_row[feature])
                except KeyError as exc:
                    raise KeyError(
                        f"CSV is missing required numeric column '{feature}'"
                    ) from exc

            for feature in CATEGORICAL_FEATURES:
                try:
                    features[feature] = normalized_row[feature]
                except KeyError as exc:
                    raise KeyError(
                        f"CSV is missing required categorical column '{feature}'"
                    ) from exc

            records.append(CustomerRecord(customer_id, features))
    return records


def _min_max(values: Sequence[float]) -> Tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        max_value = min_value + 1.0
    return min_value, max_value


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Compute a percentile from a pre-sorted sequence."""

    if not sorted_values:
        return 0.0
    index = fraction * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = index - lower
    return lower_value * (1 - weight) + upper_value * weight


def display_basic_overview(records: Sequence[CustomerRecord], *, head: int = 5) -> None:
    """Print head, info, and describe style summaries for the dataset."""

    print("First rows of the dataset:")
    headers = [ID_FIELD] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    print(", ".join(headers))
    for record in records[:head]:
        row = [str(record.customer_id)] + [
            f"{record.features[feature]}" for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ]
        print(", ".join(row))

    print("\nDataset info:")
    total = len(records)
    print(f"Total records: {total}")
    dtype_map: Dict[str, str] = {}
    dtype_map[ID_FIELD] = "int"
    for feature in NUMERIC_FEATURES:
        dtype_map[feature] = "float"
    for feature in CATEGORICAL_FEATURES:
        dtype_map[feature] = "str"
    for feature in [ID_FIELD] + NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        unique_values = {
            (record.customer_id if feature == ID_FIELD else record.features[feature])
            for record in records
        }
        print(f" - {feature} ({dtype_map[feature]}): {len(unique_values)} unique values")

    print("\nNumeric summary statistics:")
    for feature in NUMERIC_FEATURES:
        values = [float(record.features[feature]) for record in records]
        sorted_values = sorted(values)
        feature_mean = mean(values)
        feature_std = pstdev(values)
        minimum = sorted_values[0]
        maximum = sorted_values[-1]
        q1 = _percentile(sorted_values, 0.25)
        median = _percentile(sorted_values, 0.5)
        q3 = _percentile(sorted_values, 0.75)
        print(
            f" - {feature}: min={minimum:.2f}, q1={q1:.2f}, median={median:.2f}, "
            f"mean={feature_mean:.2f}, q3={q3:.2f}, max={maximum:.2f}, std={feature_std:.2f}"
        )


def preprocess_data(records: Sequence[CustomerRecord]) -> PreprocessedData:
    """Normalize numeric features and one-hot encode categoricals."""

    numeric_stats: Dict[str, Tuple[float, float]] = {}
    for feature in NUMERIC_FEATURES:
        numeric_stats[feature] = _min_max(
            [float(record.features[feature]) for record in records]
        )

    categorical_values: Dict[str, List[str]] = {}
    for feature in CATEGORICAL_FEATURES:
        distinct = sorted({str(record.features[feature]) for record in records})
        categorical_values[feature] = distinct

    feature_names: List[str] = []
    feature_names.extend(f"norm_{name}" for name in NUMERIC_FEATURES)
    for feature, values in categorical_values.items():
        feature_names.extend(f"{feature}__{value}" for value in values)

    feature_matrix: List[List[float]] = []
    customer_ids: List[int] = []

    for record in records:
        vector: List[float] = []
        for feature in NUMERIC_FEATURES:
            raw_value = float(record.features[feature])
            min_value, max_value = numeric_stats[feature]
            normalized = (raw_value - min_value) / (max_value - min_value)
            vector.append(normalized)

        for feature, values in categorical_values.items():
            record_value = str(record.features[feature])
            for value in values:
                vector.append(1.0 if record_value == value else 0.0)

        feature_matrix.append(vector)
        customer_ids.append(record.customer_id)

    return PreprocessedData(customer_ids, feature_names, feature_matrix)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

Vector = Sequence[float]
DistanceMetric = Callable[[Vector, Vector], float]


def euclidean_distance(a: Vector, b: Vector) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a: Vector, b: Vector) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def cosine_distance(a: Vector, b: Vector) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if math.isclose(norm_a, 0.0) or math.isclose(norm_b, 0.0):
        return 0.0
    cosine_similarity = dot_product / (norm_a * norm_b)
    return 1.0 - cosine_similarity


def compute_distance_matrix(
    vectors: Sequence[Vector], metric: DistanceMetric
) -> List[List[float]]:
    size = len(vectors)
    matrix: List[List[float]] = [[0.0] * size for _ in range(size)]
    for i in range(size):
        matrix[i][i] = 0.0
        for j in range(i + 1, size):
            distance = metric(vectors[i], vectors[j])
            matrix[i][j] = distance
            matrix[j][i] = distance
    return matrix


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def find_similar_customers(
    customer_id: int,
    distance_matrix: Sequence[Sequence[float]],
    customer_ids: Sequence[int],
    top_n: int = 3,
) -> List[Tuple[int, float]]:
    """Return the closest customers according to the provided matrix."""

    if customer_id not in customer_ids:
        raise ValueError(f"Customer {customer_id} not found in dataset")

    index = customer_ids.index(customer_id)
    distances = [
        (other_id, distance_matrix[index][idx])
        for idx, other_id in enumerate(customer_ids)
        if other_id != customer_id
    ]
    distances.sort(key=lambda item: item[1])
    return distances[:top_n]


def save_distance_tables(
    output_dir: Path,
    customer_ids: Sequence[int],
    distance_matrices: Dict[str, Sequence[Sequence[float]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    header = ["CustomerID"] + [str(cid) for cid in customer_ids]

    for name, matrix in distance_matrices.items():
        path = output_dir / f"{name}_distances.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for customer_id, row in zip(customer_ids, matrix):
                writer.writerow([customer_id] + [f"{value:.4f}" for value in row])


def create_html_heatmap(
    path: Path,
    customer_ids: Sequence[int],
    distance_matrices: Dict[str, Sequence[Sequence[float]]],
) -> None:
    """Generate a lightweight HTML heatmap for the distance matrices."""

    path.parent.mkdir(parents=True, exist_ok=True)

    sections: List[str] = []
    for name, matrix in distance_matrices.items():
        all_values = [value for row in matrix for value in row]
        max_value = max(all_values) if all_values else 1.0
        rows_html: List[str] = []
        for customer_id, row in zip(customer_ids, matrix):
            cells: List[str] = []
            for value in row:
                intensity = 255 - int(200 * (value / max_value)) if max_value else 255
                intensity = max(55, min(255, intensity))
                color = f"rgb({intensity}, {intensity}, 255)"
                cells.append(
                    f'<td style="background-color: {color}; text-align: right; padding: 4px;">'
                    f"{value:.3f}</td>"
                )
            rows_html.append(
                f"<tr><th style='text-align:left;padding:4px;'>Customer {customer_id}</th>"
                + "".join(cells)
                + "</tr>"
            )
        header_cells = "".join(
            f"<th style='padding:4px;'>C{cid}</th>" for cid in customer_ids
        )
        sections.append(
            f"<h2>{name.title()} distance</h2>"
            "<table border='1' cellspacing='0' cellpadding='2'>"
            "<tr><th></th>" + header_cells + "</tr>"
            + "".join(rows_html)
            + "</table>"
        )

    html = (
        "<html><head><meta charset='utf-8'><title>Customer Distance Heatmaps"\
        "</title></head><body>"
        "<h1>Customer Distance Heatmaps</h1>"
        + "".join(sections)
        + "</body></html>"
    )

    path.write_text(html, encoding="utf-8")


def build_hybrid_distance(
    distance_matrices: Dict[str, Sequence[Sequence[float]]],
    weights: Dict[str, float] | None = None,
) -> List[List[float]]:
    """Combine distance matrices using a weighted average."""

    if not distance_matrices:
        return []

    metric_names = list(distance_matrices.keys())
    size = len(distance_matrices[metric_names[0]])
    weights = weights or {name: 1.0 for name in metric_names}

    weight_sum = sum(weights.get(name, 0.0) for name in metric_names)
    if math.isclose(weight_sum, 0.0):
        raise ValueError("Weights must sum to a non-zero value")

    combined: List[List[float]] = [[0.0] * size for _ in range(size)]
    for name in metric_names:
        matrix = distance_matrices[name]
        weight = weights.get(name, 0.0)
        for i in range(size):
            for j in range(size):
                combined[i][j] += matrix[i][j] * weight

    for i in range(size):
        for j in range(size):
            combined[i][j] /= weight_sum
    return combined


def create_similarity_report(
    path: Path,
    customer_ids: Sequence[int],
    distance_matrices: Dict[str, Sequence[Sequence[float]]],
    highlight_customers: Iterable[int],
    top_n: int = 3,
) -> None:
    """Write a human-readable report of similar customers for each metric."""

    lines: List[str] = []
    for customer_id in highlight_customers:
        lines.append(f"Customer {customer_id} similarity summary:")
        for name, matrix in distance_matrices.items():
            neighbours = find_similar_customers(customer_id, matrix, customer_ids, top_n)
            neighbour_text = ", ".join(
                f"Customer {other} (distance={dist:.3f})" for other, dist in neighbours
            )
            lines.append(f"  - {name.title()}: {neighbour_text}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main routine tying everything together
# ---------------------------------------------------------------------------

def main() -> None:
    data_path = Path("retail_customers.csv")
    records = load_customer_data(data_path)
    display_basic_overview(records)

    processed = preprocess_data(records)

    metrics: Dict[str, DistanceMetric] = {
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "cosine": cosine_distance,
    }

    distance_matrices: Dict[str, List[List[float]]] = {}
    for name, metric in metrics.items():
        distance_matrices[name] = compute_distance_matrix(
            processed.feature_matrix, metric
        )

    hybrid_matrix = build_hybrid_distance(distance_matrices)
    distance_matrices["hybrid"] = hybrid_matrix

    save_distance_tables(Path("outputs"), processed.customer_ids, distance_matrices)
    create_html_heatmap(
        Path("figures") / "distance_heatmaps.html",
        processed.customer_ids,
        distance_matrices,
    )

    report_customers = processed.customer_ids[:5]
    create_similarity_report(
        Path("outputs") / "similarity_report.txt",
        processed.customer_ids,
        distance_matrices,
        report_customers,
    )


if __name__ == "__main__":
    main()
