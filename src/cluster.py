import argparse
import json
import logging
import textwrap
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from numpy import ndarray
from scipy import sparse
from scipy.sparse import spmatrix
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="7")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default="data/data.pkl",
        help="Path to input pickle file from `data.py`.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="data/data_clusters.pkl",
        help="Path to save CSV with cluster labels. (If omitted, results are not written.)",
    )
    parser.add_argument(
        "--k-range",
        nargs=3,
        type=int,
        metavar=("START", "END", "STEP"),
        help="Sweep k from START to END-1 in increments of STEP (e.g. 5 35 5 → 5,10,…,30).",
        # default=[100, 1001, 100],
        default=None,
    )
    parser.add_argument(
        "--n-clusters",
        # type=int,
        default=1000,
        help="k for one-shot clustering when --k-range not supplied.",
    )
    parser.add_argument(
        "--random-state", type=int, default=69, help="Random seed for reproducibility."
    )

    # Parallelism knob
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=6,
        help="Number of parallel processes for the k sweep (default -1 = all cores).",
    )

    # Vectoriser knobs
    parser.add_argument("--max-title-features", type=int, default=10_000)
    parser.add_argument("--min-title-df", type=int, default=100)
    parser.add_argument("--max-title-ngram", type=int, default=3)
    parser.add_argument("--max-desc-features", type=int, default=3_000)
    parser.add_argument("--min-desc-df", type=int, default=100)
    parser.add_argument("--max-desc-df", type=int, default=0.5)
    parser.add_argument("--max-desc-ngram", type=int, default=3)
    parser.add_argument("--max-tag-features", type=int, default=1_000)
    parser.add_argument("--min-df-tag", type=int, default=100)

    parser.add_argument(
        "--features",
        "-f",
        nargs="+",
        choices=["video_title", "desc", "tags", "cat"],
        default=["video_title"],
        help="Which features to include in the ColumnTransformer. "
        "Options: video_title, desc, tags, cat.",
    )

    return parser.parse_args()


def drop_zero_rows(X: ndarray | spmatrix) -> ndarray | spmatrix:
    if sparse.issparse(X):
        # getnnz(axis=1) counts nonzero entries in each row
        nonzero_mask = X.getnnz(axis=1) != 0  # True for rows we want to keep
        # either boolean‐index or fancy‐index the rows:
        try:
            return X[nonzero_mask]  # CSR supports boolean mask on rows
        except (TypeError, IndexError):
            # fallback to integer indexing
            keep_idx = np.nonzero(nonzero_mask)[0]
            return X[keep_idx]
    else:
        # ensure it’s an array
        X = np.asarray(X)
        # any nonzero in the row → keep
        nonzero_mask = np.any(X != 0, axis=1)
        return X[nonzero_mask]


def write_cluster_report(
    df: pd.DataFrame,
    features: ColumnTransformer,
    km: KMeans,
    output_path: str,
    sample_size: int = 5,
    wrap_width: int = 80,
) -> None:
    feature_names = features.get_feature_names_out()
    centroids = km.cluster_centers_

    # Compute sizes for all clusters (including empty ones)
    size_series = df["cluster"].value_counts().to_dict()
    n_clusters = centroids.shape[0]
    cluster_ids = list(range(n_clusters))
    # Sort clusters by size descending
    cluster_ids.sort(key=lambda x: size_series.get(x, 0), reverse=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Cluster Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        for i in cluster_ids:
            members = df[df["cluster"] == i]
            n_members = len(members)

            header = f"Cluster {i} ({n_members} items)"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            top_idx = centroids[i].argsort()[::-1][:10]
            top_terms = feature_names[top_idx]
            f.write("Top Terms:\n")
            f.write("  " + ", ".join(top_terms) + "\n\n")

            f.write("Examples:\n")
            if n_members == 0:
                f.write("  (No members in this cluster)\n\n")
            else:
                samples = members[["video_title", "description", "tags_str"]].sample(
                    min(sample_size, n_members), random_state=42
                )
                for _, row in samples.iterrows():
                    title = row.video_title.strip() or "[No title]"
                    desc = (
                        (row.description.strip()[:100] + "...")
                        if len(row.description.strip()) > 100
                        else (row.description.strip() or "[No description]")
                    )
                    tags = row.tags_str.strip()[:30] or "[No tags]"

                    wrapped_title = textwrap.fill(title, wrap_width)
                    wrapped_desc = textwrap.fill(desc, wrap_width)
                    wrapped_tags = textwrap.fill(tags, wrap_width)

                    f.write(f"  • Title: {wrapped_title}\n")
                    f.write(f"    Description: {wrapped_desc}\n")
                    f.write(f"    Tags: {wrapped_tags}\n\n")

            f.write("=" * 80 + "\n\n")

    logger.info(f"✔ Cluster report saved to: {output_path}")


def build_feature_matrix(
    df: pd.DataFrame, args
) -> tuple[ndarray | spmatrix, ColumnTransformer]:
    """Returns (X sparse matrix, fitted feature transformer)."""
    title_tf = TfidfVectorizer(
        max_features=args.max_title_features,
        ngram_range=(1, args.max_title_ngram),
        stop_words="english",
        min_df=args.min_title_df,
    )
    desc_tf = TfidfVectorizer(
        max_features=args.max_desc_features,
        ngram_range=(1, args.max_desc_ngram),
        stop_words="english",
        min_df=args.min_desc_df,
        max_df=args.max_desc_df,
    )
    tags_tf = TfidfVectorizer(
        max_features=args.max_tag_features,
        min_df=args.min_df_tag,
        token_pattern=r"[^ ]+",  # split on spaces only
    )
    cat_ohe = OneHotEncoder(handle_unknown="ignore")

    all_features = [
        ("video_title", title_tf, "video_title"),
        ("desc", desc_tf, "description"),
        ("tags", tags_tf, "tags_str"),
        ("cat", cat_ohe, ["category"]),
    ]
    # filter by what the user asked for
    selected = [feat for feat in all_features if feat[0] in args.features]
    if not selected:
        raise ValueError(
            "No features selected! Please choose at least one of: "
            "video_title, desc, tags, cat."
        )

    features = ColumnTransformer(selected, sparse_threshold=0.3)

    X = features.fit_transform(df)
    X = normalize(X, norm="l2", axis=1, copy=False)
    return X, features  # type: ignore


def _init_worker(logfile: str | None = None) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = (
        logging.StreamHandler() if logfile is None else logging.FileHandler(logfile)
    )
    fmt = "%(asctime)s [%(processName)s] %(levelname)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _cluster_for_k(k: int, X, random_state: int):
    """Helper to fit KMeans and compute silhouette in a subprocess. Logs timing."""
    start_t = time.perf_counter()

    km = KMeans(n_clusters=k, random_state=random_state)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, metric="cosine")

    fit_seconds = time.perf_counter() - start_t
    minutes, seconds = divmod(fit_seconds, 60)
    logging.info(
        f"k={k:<3d}; sil={sil:.4f}; recon_error={km.inertia_:,.0f}; fitted in {int(minutes)}m {seconds:.2f}s"
    )

    return k, sil, km.inertia_, labels, fit_seconds, km


def make_clusters(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, KMeans, ColumnTransformer]:
    """Load data, build feature pipeline, fit clustering model, return DataFrame."""
    df: pd.DataFrame = pd.read_pickle(args.input)

    # Text columns
    df = df[df["video_title"].notna() & (df["video_title"].str.strip() != "")]
    df["description"] = df["description"].fillna("")

    # tags: convert list → space‑separated string
    df["tags_str"] = df["tags"].apply(
        lambda lst: " ".join(lst) if isinstance(lst, list) else ""
    )

    # categories: use first element of list (YouTube gives one per video)
    df["category"] = df["categories"].apply(
        lambda lst: lst[0] if isinstance(lst, list) and lst else "Unknown"
    )

    X, features = build_feature_matrix(df, args)
    logger.info(f"Created feature vectors with shape={X.shape}")
    if sparse.issparse(X):
        zero_count = int((X.getnnz(axis=1) == 0).sum())
    else:
        X = np.asarray(X)
        zero_count = int(np.all(X == 0, axis=1).sum())
    logger.info(f"Number of zero feature vectors: {zero_count}")

    # X = drop_zero_rows(X)
    # if sparse.issparse(X):
    #     zero_count = int((X.getnnz(axis=1) == 0).sum())
    # else:
    #     X = np.asarray(X)
    #     zero_count = int(np.all(X == 0, axis=1).sum())
    # logger.info(f"Number of zero feature vectors: {zero_count}")

    k_values = list(range(*args.k_range)) if args.k_range else [args.n_clusters]

    # Run the sweep in parallel
    results = Parallel(
        n_jobs=args.n_jobs,
        prefer="processes",
        verbose=5,
        initializer=_init_worker,
        initargs=(None,),
    )(delayed(_cluster_for_k)(k, X, args.random_state) for k in k_values)

    # Sort results by k to keep plots tidy
    results = list(results)
    results.sort(key=lambda x: x[0])

    sil_scores, recon_errors = [], []
    best_k, best_sil, best_labels, best_km = None, -1, None, None

    for k, sil, inertia, labels, _, km in results:
        sil_scores.append(sil)
        recon_errors.append(inertia)
        if sil > best_sil:
            best_k, best_sil, best_labels, best_km = k, sil, labels, km

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_ts = Path("data/experiments") / timestamp
    out_dir_ts.mkdir(parents=True, exist_ok=True)

    if args.k_range:
        plot_path = out_dir_ts / f"{timestamp}_silhouette.png"

        plt.figure()
        plt.plot(k_values, sil_scores, marker="o")
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Silhouette score (cosine distance)")
        plt.title("Silhouette vs. k")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"Silhouette plot → {plot_path}")

        recon_path = out_dir_ts / f"{timestamp}_recon.png"
        plt.figure()
        plt.plot(k_values, recon_errors, marker="o")
        plt.xlabel("k")
        plt.ylabel("Reconstruction error (inertia)")
        plt.title("Reconstruction error vs. k")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(recon_path)
        logger.info(f"Reconstruction error plot → {recon_path}")

    args_json = out_dir_ts / "args.json"
    with args_json.open("w") as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            f,
            indent=2,
        )

    df["cluster"] = best_labels
    if args.output:
        df.to_pickle(args.output)
        logger.info(f"Clustered data (best k={best_k}) written to → {args.output}")
        dump(features, out_dir_ts / "features.joblib")
        dump(best_km, out_dir_ts / "kmeans.joblib")
        logger.info(f"best_km and features written to → {out_dir_ts}")

        if hasattr(X, "toarray"):
            X = X.toarray()
        else:
            X = np.asarray(X)

        cluster_sizes = Counter(labels)
        top_clusters = [cid for cid, _ in cluster_sizes.most_common(50)]
        mask = np.isin(labels, top_clusters)
        X = X[mask]
        labels = labels[mask]

        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap("nipy_spectral", len(top_clusters))

        sc = plt.scatter(
            coords[:, 0],
            coords[:, 1],
            c=[top_clusters.index(l) for l in labels],
            cmap=cmap,
            s=5,
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA projection of clusters")

        cbar = plt.colorbar(
            sc, ticks=range(len(top_clusters)), fraction=0.046, pad=0.04
        )
        cbar.ax.set_yticklabels(top_clusters)
        cbar.set_label("Cluster ID", rotation=270, labelpad=15)

        pca_path = out_dir_ts / "pca.png"
        plt.savefig(pca_path)
        logger.info(f"saved PCA plot → {pca_path}")

        write_cluster_report(
            df, features, best_km, output_path=str(out_dir_ts / "cluster_report.txt")
        )

    return df, best_km, features  # type: ignore


if __name__ == "__main__":
    logger.info("Start")
    cli_args = get_args()
    make_clusters(cli_args)
