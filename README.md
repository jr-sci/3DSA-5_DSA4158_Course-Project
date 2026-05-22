# 3DSA-5_DSA4158_Course-Project
# Integration of Word2Vec-Based Sequential Clustering and PageRank Topological Analysis for Identifying Authority Hubs in Amazon Product Metadata

A PySpark pipeline that combines **sequence-based product embeddings** with **graph-based influence scoring** to identify *authority hubs* which are products that are not just popular, but structurally influential in the Amazon Home & Kitchen 2023 dataset.

---

## Overview

Most "top product" rankings on e-commerce platforms reduce to raw popularity (review count or sales rank). This project argues that **popularity ≠ influence**, and tests whether the two diverge in a measurable way.

The pipeline aims to answers three questions:

1. **Behavioral grouping**
   - Do co-purchase sequences embed products into meaningful behavioral clusters? 
2. **Structural influence**
   - Which products act as authority hubs in the co-review graph? 
3. **Hub concentration**
   - Are high-PageRank products distributed uniformly across behavioral clusters, or do they concentrate in specific ones? 

The output is a merged dataset linking each product to its behavioral cluster, its PageRank score, its popularity, and its rating.

---

## Dataset

- **Source**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD)
- **Category**: Home and Kitchen
- **Files used**:
  - `meta_Home_and_Kitchen.jsonl.gz` — product metadata
  - `Home_and_Kitchen.jsonl.gz` — user reviews

The notebook downloads both files automatically, converts them to Parquet, and caches results on Google Drive.

---

## Requirements

The notebook is built for **Google Colab with Drive** mounted. 

Key dependencies:

```
pyspark
networkx
pyarrow
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
tqdm
requests
```

> [!TIP]
> **Recommended runtime**: Colab with high-RAM enabled. Spark is configured with `spark.driver.memory = 12g` and `spark.driver.maxResultSize = 2g`.

---

## Setup

1. Open the notebook in Google Colab.
2. In your Google Drive, create the folder:
   ```
   MyDrive/Amazon/DATASETS
   ```
   (or edit `DRIVE_DIR` in the **Load Dataset** cell to point elsewhere).
3. Run the cells top to bottom. The first full run will:
   - Download ~10–15 GB of raw data
   - Convert it to Parquet on Drive
   - Train Word2Vec, run K-Means, build the co-review graph, and run PageRank
   - Cache every intermediate result so subsequent runs skip straight to analysis.

> [!NOTE]
> Expect the first end-to-end run to take **several hours**.

> [!TIP]
> The co-review edge threshold (`weight >= 3`) and the top-N cap (`5000`) are tuning knobs. Lowering them grows the graph and PageRank runtime significantly.
---

## Pipeline structure

| Section | What it does | Output written to Drive |
|---|---|---|
| Preliminary Setup | Imports, Spark session, Drive mount | — |
| Dataset Configuration | Download + JSONL → Parquet | `meta_Home_and_Kitchen.parquet`, `Home_and_Kitchen.parquet` |
| Feature Extraction | Build user sequences, train Word2Vec | `user_sequences.parquet`, `embeddings.parquet` |
| PageRank | Build co-review graph, run PageRank | `pagerank_co_review.parquet` |
| K-Means Clustering | Cluster the Word2Vec embeddings | `clusters.parquet` |
| Merge & Validation | Join all signals, run statistical tests | `final_analysis.parquet`, `cluster_summary.parquet` |
| Visualizations | 7 plots + summary report | `outputs/summary_statistics.txt` |

---

## File layout after a full run

```
MyDrive/Amazon/DATASETS/
├── meta_Home_and_Kitchen.parquet/      # product metadata
├── Home_and_Kitchen.parquet/           # reviews
├── user_sequences.parquet/             # timestamp-ordered purchase sequences
├── embeddings.parquet/                 # 64-d Word2Vec vectors per product
├── clusters.parquet/                   # K-Means assignments (k=50)
├── pagerank_co_review.parquet/         # PageRank scores + metadata
├── final_analysis.parquet/             # merged dataset
├── cluster_summary.parquet/            # per-cluster aggregates
└── outputs/
    └── summary_statistics.txt          # text summary report
```
> [!NOTE]
> All Drive paths assume `MyDrive/Amazon/DATASETS`. Update `DRIVE_DIR` if your layout differs.
---

## License

Dataset usage is governed by the [Amazon Reviews 2023 terms](https://amazon-reviews-2023.github.io/). 
