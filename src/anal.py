import argparse
import logging
import pandas as pd

from views import create_views_animations


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
mpl_logger = logging.getLogger("matplotlib.category")
mpl_logger.setLevel(logging.WARNING) 
logger = logging.getLogger(__name__)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_filepath", help="Path to the input pickle file created from `data.py`")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    df: pd.DataFrame = pd.read_pickle(args.pickle_filepath)

    logger.info("Started creatiing view time histogram animations")
    create_views_animations(df=df.copy())

