import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yt_dlp
from tqdm.auto import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath",
        help="Path to the input JSON file (e.g. data/história pozerania.json)",
    )
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--output_filepath", type=str, default="data/data.pkl")
    args = parser.parse_args()
    return args


def get_df(args: argparse.Namespace) -> pd.DataFrame:
    with open(args.filepath, "r") as f:
        data = json.load(f)
    logger.info("Loaded %d records from %s", len(data), args.filepath)

    df = pd.json_normalize(data, errors="ignore")
    df["time"] = pd.to_datetime(
        df["time"],
        format="mixed",
        utc=True,  # parse Z as UTC
    )

    df = df[df["details"].isna()]
    df.pop("details")

    for col in ["products", "header", "activityControls", "subtitles", "description"]:
        df.pop(col)

    return df


def fetch_info_record(args: tuple) -> tuple[int, dict]:
    opts = {"quiet": True, "skip_download": True}
    idx, url = args
    if not url or pd.isna(url):
        return idx, {
            "video_title": None,
            "categories": None,
            "tags": None,
            "description": None,
        }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info is None:
            return idx, {
                "video_title": None,
                "categories": None,
                "tags": None,
                "description": None,
            }
    return idx, {
        "video_title": info.get("title"),
        "categories": info.get("categories"),
        "tags": info.get("tags"),
        "description": info.get("description"),
    }


def enrich_metadata(df: pd.DataFrame, max_workers: int) -> pd.DataFrame:
    tasks = [(idx, url) for idx, url in df["titleUrl"].dropna().items()]
    results = []
    failed_idxs = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_info_record, t): t[0] for t in tasks}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Enriching video metadata"
        ):
            try:
                idx, record = future.result()
            except Exception as e:
                idx = futures[future]
                failed_idxs.append(idx)
                logger.exception(f"Unexpected error for idx={idx}: {e}")
                record: dict = {
                    "video_title": None,
                    "categories": None,
                    "tags": None,
                    "description": None,
                }
            record["idx"] = idx
            results.append(record)

    info_df = pd.DataFrame(results).set_index("idx")
    info_df = info_df.astype("object")

    df = df.join(info_df)

    n_fail = len(failed_idxs)
    if n_fail:
        logger.warning(f"Completed with {n_fail} failures.")
        failed_path = "data/failed_idxs.json"
        with open(failed_path, "w") as f:
            json.dump(failed_idxs, f)
        logger.info(f"Saved failed idxs to {failed_path}")

    return df


if __name__ == "__main__":
    args = get_args()
    max_workers = args.max_workers

    df = get_df(args)
    df = enrich_metadata(df, max_workers)

    df.to_pickle(args.output_filepath)
    logger.info(f"Dataframe saved to {args.output_filepath}")
