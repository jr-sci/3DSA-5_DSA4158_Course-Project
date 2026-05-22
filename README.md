# 3DSA-5_DSA4158_Course-Project
# Integration of Word2Vec-Based Sequential Clustering and PageRank Topological Analysis for Identifying Authority Hubs in Amazon Product Metadata

A PySpark pipeline that combines **sequence-based product embeddings** (Word2Vec on user purchase histories) with **graph-based influence scoring** (PageRank on a co-review network) to identify *authority hubs* — products that are not just popular, but structurally influential — in the Amazon Home & Kitchen 2023 dataset.

---

## Overview

Most "top product" rankings on e-commerce platforms reduce to raw popularity (review count or sales rank). This project argues that **popularity ≠ influence**, and tests whether the two diverge in a measurable way.

The pipeline answers three questions:

1. **Behavioral grouping** — Do co-purchase sequences embed products into meaningful behavioral clusters? *(Word2Vec → K-Means)*
2. **Structural influence** — Which products act as authority hubs in the co-review graph? *(PageRank on the co-review network)*
3. **Hub concentration** — Are high-PageRank products distributed uniformly across behavioral clusters, or do they concentrate in specific ones? *(Chi-square test + Spearman correlation)*

The output is a merged dataset linking each product to its behavioral cluster, its PageRank score, its popularity, and its rating — plus a set of validation tests and seven visualizations.

---

## Methodology

### 1. Word2Vec on purchase sequences
Each user's reviewed products are ordered by timestamp to form a "sentence." Word2Vec (Spark MLlib) learns a 64-dimensional embedding per product such that products frequently bought in similar sequential contexts end up near each other in vector space.

- Filter: users with 3–100 reviews (removes bots and one-off accounts)
- Vector size: 64, minCount: 10, maxIter: 5

### 2. K-Means clustering
The learned product vectors are partitioned into **k = 50** behavioral clusters using Spark MLlib's K-Means (seed = 42, maxIter = 20). Clusters represent behaviorally-coherent product groups derived from real co-purchase patterns, not category labels.

### 3. PageRank on the co-review graph
- **Nodes**: the top 5,000 most-reviewed products
- **Edges**: two products are connected if ≥ 3 distinct users reviewed both
- **Algorithm**: NetworkX PageRank with α = 0.85, tol = 1e-6, max_iter = 50

High PageRank → the product sits at a structural crossroads of user attention.

### 4. Validation
- **Chi-square test** — are influential products (top 20% by PageRank) distributed uniformly across the 50 behavioral clusters?
- **Spearman correlation** — how strongly does PageRank correlate with raw popularity and average rating?
- **Cluster drill-down** — manual inspection of the top 3 high-influence clusters.

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

The notebook is built for **Google Colab** with Drive mounted. Key dependencies:

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

PySpark is installed inline via `!pip install pyspark`. Everything else is preinstalled on Colab.

**Recommended runtime**: Colab with high-RAM enabled. Spark is configured with `spark.driver.memory = 12g` and `spark.driver.maxResultSize = 2g`.

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

Expect the first end-to-end run to take **several hours**. Re-runs that hit cached Parquet files take minutes.

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

## Visualizations produced

1. **Cluster Size vs Influence** — scatter of cluster size against average PageRank, sized by popularity, colored by rating.
2. **PageRank Distribution Across Clusters** — boxplot across the top 20 clusters.
3. **Influence Concentration Heatmap** — normalized cluster characteristics for the top 25.
4. **PageRank vs Popularity** — log-log scatter colored by cluster, showing where influence diverges from popularity.
5. **Influence Concentration by Cluster** — horizontal bar chart of the 20 clusters with the highest share of top-PageRank products.
6. **Feature Correlation Matrix** — Spearman correlations between PageRank, review count, and rating.
7. **Top Products per High-Influence Cluster** — tables of the leading products in the three most influential clusters.

---

## Key findings 

Concrete numbers vary by run, but the pipeline reports:

- A **Spearman correlation between PageRank and review count** — interpreted as: if < 0.7, PageRank is capturing influence that pure popularity misses.
- A **chi-square statistic** on the cluster-wise distribution of influential products — testing whether authority hubs concentrate in specific behavioral clusters rather than spreading uniformly.
- The **top 3 high-influence clusters**, drilled down to show which products anchor them.

See `outputs/summary_statistics.txt` after a run for the actual numbers.

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

---

## Notes

- The co-review edge threshold (`weight >= 3`) and the top-N cap (`5000`) are tuning knobs. Lowering them grows the graph and PageRank runtime significantly.
- The Word2Vec hyperparameters in the running code (`vectorSize=64`, `minCount=10`, `maxIter=5`) differ from the print statement above the cell (`128 / 5 / 10`); the cell's actual arguments take precedence.
- PageRank runs in-memory via NetworkX after collecting edges to the driver. For graphs much larger than the top-5000 subgraph, swap in GraphFrames or pyspark-graphframes PageRank.
- All Drive paths assume `MyDrive/Amazon/DATASETS`. Update `DRIVE_DIR` if your layout differs.

---

## License

Dataset usage is governed by the [Amazon Reviews 2023 terms](https://amazon-reviews-2023.github.io/). 
