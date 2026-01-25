# WatchUp

WatchUp is a small Flask application that exposes a web interface for downloading media with [yt-dlp](https://github.com/yt-dlp/yt-dlp). It also includes a lightweight podcast workflow that turns uploaded `.txt`/`.md` files into OpenAI-generated scripts and TTS audio. The project focuses on personal utility: track download progress, log history, and store outputs in predictable local folders.

## Key Features

- **Web UI for yt-dlp**: Submit a video URL, choose quality, audio-only/video, optional subtitles, and clip sections.
- **Background downloads + live progress**: Each download runs in a background thread, streaming stdout to a per-job log file that the UI polls.
- **Download history**: Every download (including magnets) is appended to a local JSON history file.
- **Magnet downloads via aria2c**: Magnet links are handed to `aria2c` with the chosen download folder.
- **Text → podcast workflow**: Upload text/markdown files to generate a scripted podcast and MP3 audio using OpenAI.
- **Extensible blueprints**: `app.py` loads an additional `shopinsight` blueprint from a sibling repo if available.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5001/yt`.

## Configuration

### `config.py`

- `DOWNLOAD_ROOT`: absolute path where downloads are stored. Defaults to `/Users/Admin/Media/Other`.
- `DOWNLOAD_FOLDERS`: mapping of display names to folders. Defaults to `Videos`, `Music`, `Clips`, all pointing at `DOWNLOAD_ROOT`.
- `DOWNLOAD_ROOT` is created on startup with `os.makedirs(..., exist_ok=True)`.

### Flask settings in `app.py`

- `app.secret_key = 'dev'`
- `MAX_CONTENT_LENGTH = 50 * 1024 * 1024` (50 MB upload limit for file uploads)
- App runs at `0.0.0.0:5001` in debug mode.

### Environment variables

- `OPENAI_API_KEY` is required for the podcast workflow.
  - It is loaded from `openai_api_key.env` in the repo root (see `load_api_key_from_env`).
  - The file must exist and contain `OPENAI_API_KEY=...`.

### Optional dependencies

- `aria2c` is required for `/yt/magnet` downloads.
- `ffmpeg` is commonly required by `yt-dlp` for remuxing video or extracting audio (see Troubleshooting).

## Route Catalog

All routes live in `__init__.py` under the `yt_bp` Flask blueprint (registered in `app.py`).

### `GET /yt`

- **Purpose**: Render the main downloader UI.
- **Parameters**: none
- **Output/side effects**: Renders `templates/yt/index.html`.
- **Code**: `__init__.py::index`.

### `POST /yt`

- **Purpose**: Start a background `yt-dlp` download.
- **Form fields**:
  - `url` (required)
  - `quality` (e.g., `best`, `bestvideo[height<=720]+bestaudio`)
  - `format` (`audio` or `video`)
  - `audio_container` (default `mp3`)
  - `video_container` (default `mp4`)
  - `subtitles` (checkbox)
  - `section` (optional clip range, e.g. `00:10:00-00:15:00`)
  - `folder` (key from `DOWNLOAD_FOLDERS`)
- **Output/side effects**:
  - Creates a background thread running `yt-dlp`.
  - Writes progress to `progress_<uid>.log` in the repo root.
  - Appends a history entry to `download_history.json`.
  - Redirects back to `/yt` with a flash message containing the job ID.
- **Code**: `__init__.py::index`, `run_download_and_track`, `log_download`.

### `GET /yt/progress/<uid>`

- **Purpose**: Return the raw contents of a progress log file.
- **Parameters**: `uid` is the job ID displayed after starting a download.
- **Output/side effects**:
  - Returns log contents or `Waiting for log...`.
- **Code**: `__init__.py::get_progress`.

### `GET /yt/history`

- **Purpose**: View download history.
- **Parameters**: none
- **Output/side effects**:
  - Renders `templates/yt/history.html`.
  - Reads `download_history.json` if it exists.
- **Code**: `__init__.py::history`.

### `GET /yt/clear_logs`

- **Purpose**: Delete all progress log files.
- **Parameters**: none
- **Output/side effects**:
  - Deletes every `progress_*.log` in the repo root.
  - Flash message with number deleted.
- **Code**: `__init__.py::clear_logs`.

### `GET /yt/clear_history`

- **Purpose**: Clear the download history JSON.
- **Parameters**: none
- **Output/side effects**:
  - Overwrites `download_history.json` with an empty list.
- **Code**: `__init__.py::clear_history`.

### `POST /yt/magnet`

- **Purpose**: Start a magnet download via `aria2c`.
- **Form fields**:
  - `magnet` (required)
  - `folder` (key from `DOWNLOAD_FOLDERS`)
- **Output/side effects**:
  - Validates `aria2c` is installed.
  - Spawns a background thread for `aria2c`.
  - Writes progress to `progress_<uid>.log`.
  - Logs history to `download_history.json`.
- **Code**: `__init__.py::magnet_download`, `run_download_and_track`, `log_magnet_download`.

### `GET /yt/podcast`

- **Purpose**: Render the podcast generation UI.
- **Parameters**: none
- **Output/side effects**: Renders `templates/yt/podcast.html`.
- **Code**: `__init__.py::podcast`.

### `POST /yt/podcast`

- **Purpose**: Generate a script + audio from uploaded text files.
- **Form fields**:
  - `files` (multiple uploads, `.txt`/`.md` only)
  - `voice` (one of `VOICE_OPTIONS`)
  - `tone` (one of `TONE_OPTIONS`)
- **Output/side effects**:
  - Saves uploads under `podcast_files/inputs/`.
  - Generates script files in `podcast_files/scripts/`.
  - Generates MP3 audio in `podcast_files/audio/`.
  - Renders results + errors in `templates/yt/podcast.html`.
- **Code**: `__init__.py::podcast`, `generate_podcast_script`, `tts_generate_audio`.

### `GET /yt/podcast/download/<category>/<path:filename>`

- **Purpose**: Download podcast artifacts as attachments.
- **Parameters**:
  - `category`: `scripts`, `audio`, or `inputs`.
  - `filename`: requested file name (must match `secure_filename`).
- **Output/side effects**: Sends a file with `Content-Disposition: attachment`.
- **Code**: `__init__.py::podcast_download`.

### `GET /yt/podcast/media/<path:filename>`

- **Purpose**: Stream generated audio files in the browser.
- **Parameters**: `filename` (must match `secure_filename`).
- **Output/side effects**: Sends the MP3 file inline.
- **Code**: `__init__.py::podcast_media`.

## Module & Function Reference

### `app.py`

- **Role**: Flask app entry point. Registers the `yt` blueprint and imports/registers a `shopinsight` blueprint from `../ShopInsight/Modules`.
- **Important notes**:
  - The import `from shopinsight import shopinsight_bp` is unconditional; if the ShopInsight module is missing, app startup will fail.

### `__init__.py`

- **Role**: Defines the `yt` blueprint, download logic, podcast workflow, and helper utilities.

Key helpers:

- `log_download(url, folder_path)`
  - Uses `yt-dlp` metadata extraction to log title + URL to `download_history.json`.
- `log_magnet_download(link, folder_path)`
  - Similar to `log_download`, but logs a generic “Magnet Download” title.
- `run_download_and_track(cmd, uid)`
  - Spawns a subprocess and streams stdout into `progress_<uid>.log`.
  - Writes the `FileNotFoundError` message into the log if the binary is missing.
- `load_api_key_from_env(env_file)`
  - Loads `OPENAI_API_KEY` from `openai_api_key.env` and raises if missing.
- `get_openai_client()`
  - Returns an OpenAI client configured with the API key.
- `is_allowed_text_file(filename)`
  - Enforces `.txt` or `.md` extensions.
- `build_podcast_prompt(tone)`
  - Constructs the system prompt for the podcast script generation.
- `generate_podcast_script(client, text, tone)`
  - Calls the OpenAI Chat Completions API with the prompt + user text.
- `tts_generate_audio(client, script_text, voice)`
  - Calls the OpenAI audio API to generate MP3 audio bytes.

### `config.py`

- **Role**: Defines absolute storage locations for downloads. Provides `DOWNLOAD_ROOT` and `DOWNLOAD_FOLDERS`.

### Templates & static assets

- `templates/yt/index.html`: Main download UI, magnet form, and progress polling.
- `templates/yt/history.html`: History list + “download again” button.
- `templates/yt/podcast.html`: Upload UI + audio playback.
- `static/style.css`: Styling shared by the UI.

## Data & Storage Layout

All paths are relative to the repo root unless otherwise noted.

- `download_history.json`
  - JSON array of objects:
    ```json
    {
      "url": "https://example.com",
      "title": "Video Title",
      "folder": "/Users/Admin/Media/Other",
      "timestamp": "2024-01-01T12:00:00.000000"
    }
    ```
  - Stored at: `<repo>/download_history.json`.
- `progress_<uid>.log`
  - Text log of stdout from `yt-dlp` or `aria2c`.
  - Stored at: `<repo>/progress_<uid>.log`.
- Downloaded media
  - Stored under `DOWNLOAD_ROOT` (default `/Users/Admin/Media/Other`) and named via `%(title)s.%(ext)s` from `yt-dlp`.
- Podcast data
  - Inputs: `podcast_files/inputs/`
  - Scripts: `podcast_files/scripts/`
  - Audio: `podcast_files/audio/`
  - All directories are created automatically on app startup.

## Security Notes

- File uploads for the podcast workflow are sanitized with `secure_filename` and restricted to `.txt`/`.md` extensions.
- Podcast download and media routes reject filenames that differ from their sanitized form, preventing path traversal.

## Troubleshooting

- **`yt-dlp` not found**: If `yt-dlp` is missing, the progress log will contain the `FileNotFoundError` message. Install `yt-dlp` and retry.
- **`aria2c` not installed**: `/yt/magnet` checks for `aria2c` and flashes an error if missing.
- **OpenAI errors**: The podcast flow requires `openai_api_key.env` with `OPENAI_API_KEY`. If the file is missing or the key is empty, the UI flashes an error.
- **Permission/path issues**: Make sure `DOWNLOAD_ROOT` exists and is writable by the user running the Flask app.
- **`ffmpeg`/remux failures**: `yt-dlp` uses `ffmpeg` for audio extraction and video remuxing. Install `ffmpeg` if you see errors like “ffprobe/ffmpeg not found.”
- **Subtitles/sections fail**: Incorrect `section` ranges or missing subtitle data can cause `yt-dlp` errors; review the progress log output for details.

## Roadmap / Planned Improvements (Suggestions)

- Add server-side validation for folder keys and format selections.
- Move `shopinsight` blueprint loading behind a try/except toggle.
- Provide a web UI for deleting individual history entries.

