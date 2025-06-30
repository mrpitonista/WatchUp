# WatchUp

WatchUp is a small Flask application that exposes a web interface for downloading media using [yt-dlp](https://github.com/yt-dlp/yt-dlp).  It provides basic download management features such as progress tracking and a simple history log.  The project is intended as a lightweight personal downloader rather than a polished product.

## Features

* **Web interface for yt-dlp** – submit a video URL, select the desired quality or audio-only mode, choose a destination folder and optional subtitle/section options.
* **Background downloads** – downloads run in a background thread while their progress is written to a log file.  A job identifier is displayed so progress can be checked from the browser.
* **Download history** – each request is logged to `download_history.json` with timestamp, title, URL and destination.
* **History/Log management** – endpoints exist to clear history or any progress log files.
* **Blueprint friendly** – the application registers a second blueprint named `shopinsight` if it is available.  This repository does not include it but `app.py` expects it to exist alongside `WatchUp` in a sibling folder `ShopInsight/Modules`.

## Requirements

* Python 3.8+
* `yt-dlp` for fetching the videos
* Flask
* Celery and Redis are listed in `requirements.txt` but are not currently used by the default code.

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Configuration

`config.py` defines where downloaded files will be saved.  By default they are placed under `DOWNLOAD_ROOT` which is currently set to `/Users/Admin/Media/Other`.  Adjust this path and the `DOWNLOAD_FOLDERS` mapping to suit your environment before running the app.

The file also contains placeholder configuration for Celery/Redis.  These settings are unused unless additional tasks are added.

## Running the application

1. Ensure Python is installed and dependencies are available.
2. Edit `config.py` to set your download directory.
3. Start the Flask server:

```bash
python app.py
```

The application listens on port `5001` by default.  Navigate to `http://localhost:5001/yt` to access the downloader.

## Repository layout

```
app.py           - Flask application entry point registering blueprints
__init__.py      - Implementation of the YouTube/yt-dlp blueprint
config.py        - Basic configuration including download folder paths
requirements.txt - Python dependencies
static/          - Optional JavaScript assets
templates/yt/    - HTML templates for the downloader and history views
```

The repository is minimal and meant for experimentation.  Pull requests or issues for enhancements are welcome.
