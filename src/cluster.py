import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from numpy import ndarray
from scipy.sparse import spmatrix
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="2")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input pickle file from `data.py`.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save CSV with cluster labels. "
        "(If omitted, results are not written.)",
    )
    parser.add_argument(
        "--k-range",
        nargs=3,
        type=int,
        metavar=("START", "END", "STEP"),
        help="Sweep k from START to END-1 in increments of STEP "
        "(e.g. 5 35 5 → 5,10,…,30).",
        default=[10, 300, 10],
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=12,
        help="k for one-shot clustering when --k-range not supplied.",
    )
    parser.add_argument(
        "--random-state", type=int, default=69, help="Random seed for reproducibility."
    )

    # Vectoriser knobs
    parser.add_argument("--max-title-features", type=int, default=1_000)
    parser.add_argument("--max-title-ngram", type=int, default=1)
    parser.add_argument("--max-desc-features", type=int, default=1_000)
    parser.add_argument("--max-desc-ngram", type=int, default=1)
    parser.add_argument("--max-tag-features", type=int, default=1_000)
    parser.add_argument("--min-df-tag", type=int, default=5)

    return parser.parse_args()


def build_feature_matrix(
    df: pd.DataFrame, args
) -> tuple[ndarray | spmatrix, ColumnTransformer]:
    """Returns (X sparse matrix, fitted feature transformer)."""
    title_tf = TfidfVectorizer(
        max_features=args.max_title_features,
        ngram_range=(1, args.max_title_ngram),
        stop_words="english",
    )
    desc_tf = TfidfVectorizer(
        max_features=args.max_desc_features,
        ngram_range=(1, args.max_desc_ngram),
        stop_words="english",
    )
    tags_tf = TfidfVectorizer(
        max_features=args.max_tag_features,
        min_df=args.min_df_tag,
        token_pattern=r"[^ ]+",  # split on spaces only
    )
    cat_ohe = OneHotEncoder(handle_unknown="ignore")

    features = ColumnTransformer(
        [
            ("video_title", title_tf, "video_title"),
            ("desc", desc_tf, "description"),
            ("tags", tags_tf, "tags_str"),
            ("cat", cat_ohe, ["category"]),
        ],
        sparse_threshold=0.3,
    )

    X = features.fit_transform(df)
    return X, features


def make_clusters(args: argparse.Namespace) -> pd.DataFrame:
    """Load data, build feature pipeline, fit clustering model, return DataFrame."""
    df: pd.DataFrame = pd.read_pickle(args.input)
    # df = df.head(60)

    # Text columns
    df["video_title"] = df["video_title"].fillna("")
    df["description"] = df["description"].fillna("")

    # tags: convert list → space‑separated string
    df["tags_str"] = df["tags"].apply(
        lambda lst: " ".join(lst) if isinstance(lst, list) else ""
    )

    # categories: use first element of list (YouTube gives one per video)
    df["category"] = df["categories"].apply(
        lambda lst: lst[0] if isinstance(lst, list) and lst else "Unknown"
    )

    X, _ = build_feature_matrix(df, args)
    logger.info(f"Created feature vectors with shape={X.shape}")

    k_values = range(*args.k_range) if args.k_range else [args.n_clusters]
    sil_scores, recon_errors = [], []
    best_k, best_sil, best_labels = None, -1, None

    for k in k_values:
        start_time = time.perf_counter()

        km = KMeans(
            n_clusters=k,
            random_state=args.random_state,
        )
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, metric="cosine")

        minutes, seconds = divmod(time.perf_counter() - start_time, 60)
        logger.info(f"Fitting time: {int(minutes)}m {seconds:.2f}s")

        recon_errors.append(km.inertia_)
        sil_scores.append(sil)
        logger.info(f"k={k:<3d} sil={sil:.4f}  recon_error={km.inertia_:,.0f}")
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, labels

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_ts = Path("data/experiments") / timestamp
    out_dir_ts.mkdir(parents=True, exist_ok=True)

    if args.k_range:
        plot_path = out_dir_ts / f"{timestamp}_silhoutte.png"

        plt.figure()
        plt.plot(list(k_values), sil_scores, marker="o")
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Silhouette score (cosine distance)")
        plt.title("Silhouette vs. k")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"Silhouette plot → {plot_path}")

        recon_path = out_dir_ts / f"{timestamp}_recon.png"
        plt.figure()
        plt.plot(list(k_values), recon_errors, marker="o")
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
        df.to_csv(args.output, index=False)
        logger.info(f"Clustered data (best k={best_k}) written to → {args.output}")

    return df


if __name__ == "__main__":
    logger.info("Start")
    cli_args = get_args()
    make_clusters(cli_args)
