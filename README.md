# youtube-history-anal

A Python project to analyze your YouTube watch history. This tool parses your Google Takeout JSON history, enriches it with video metadata, clusters viewed videos by content features, and generates time-based animations of your watching patterns.

## Features

* **Data Extraction & Enrichment**: Parse YouTube watch history JSON and enrich each entry with metadata (title, description, tags, category) using `yt-dlp`.
* **Clustering & Reporting**: Apply TF-IDF vectorization on video titles, descriptions, tags, and categories. Perform K-Means clustering, determine the optimal number of clusters, and generate detailed reports and PCA plots.
* **Temporal Visualizations**: Create animated histograms of viewing activity by hour, day, and year, and export animations as MP4 files.

## Requirements

A `requirements.txt` file is included in this repository with all necessary dependencies. To install them, run:

```bash
pip install -r requirements.txt
```
This project was developed with Python 3.13.2.

## Project Structure

```
youtube-history-anal
├── data/                     # Outputs: pickles, clusters, animations
├── src/                      # Source code modules
│   ├── data.py               # Load and enrich history data
│   ├── cluster.py            # Feature extraction and clustering
│   ├── views.py              # Generate time-based animations
│   └── anal.py               # Orchestrates data processing, views, and clustering
├── .gitignore
└── README.md                 # This file
```

## Usage

1. **Download Your YouTube Watch History**

   1. Go to [Google Takeout](https://takeout.google.com) and select only **YouTube and YouTube Music** under your account data.
   2. In the **Multiple formats** section for YouTube history, choose **JSON**.
   3. Click **Next step** and then **Create export**. Once it’s ready, download the ZIP archive.
   4. Unzip the archive and locate the JSON file for watch history. The exact folder and file names will vary by your Google account language. For example, in Slovak it may be:

      ```
      Takeout/YouTube a YouTube Music/história/história pozerania.json
      ```

2. **Extract & Enrich Data**

   Run the data module to parse your YouTube history JSON and enrich with metadata:

   ```bash
   python src/data.py "path/to/Takeout/Watch history.json" \
       --max_workers 4 \
       --output_filepath data/data.pkl
   ```

   * `json_filepath`: Path to the YouTube history JSON file from Google Takeout (wrap in quotes if the path contains spaces).
   * `--max_workers`: Number of parallel processes for metadata fetching (default 10).
   * `--output_filepath`: Destination file for the enriched DataFrame (default `data/data.pkl`).

3. **Generate Animations & Clusters**

   Run the main analysis script to create viewing-pattern animations and cluster the videos:

   ```bash
   python src/anal.py \
       --input data/data.pkl
   ```

   This will:

   * Produce animated MP4 files in `data/`:

     * `hourly_views.mp4`
     * `daily_views.mp4`
     * `yearly_views.mp4`
   * Generate clustering results and reports (see next step).
     * `data/experiments/<timestamp>/pca.png`: PCA visualization of clustering.
     * `data/experiments/<timestamp>/cluster_report.txt`: txt file with 5 samples from each cluster and most important features of that cluster.

4. **Customize & Run Clustering Only (Optional)**

   To run clustering independently or tweak parameters:

   ```bash
   python src/cluster.py \
       -i data/data.pkl \
       -o data/data_clusters.pkl \
       --k-range 2 10 1 \
       --features video_title desc tags cat \
       --n-jobs 6
   ```

   * `--k-range`: Minimum, maximum, and step for number of clusters (e.g., `2 10 1`).
   * `--features`: List of features to include (`video_title`, `desc`, `tags`, `cat`).
   * `--n-jobs`: Number of parallel jobs for clustering sweep.

   Outputs:

   * `data/data_clusters.pkl`: DataFrame with cluster labels saved as a pickle.
   * `data/cluster_report.txt`: Detailed report and evaluation.
   * `data/pca_plot.png`: PCA visualization of clusters.
