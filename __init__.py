from flask import Blueprint, render_template, request, redirect, url_for, flash
import subprocess
import os
from config import DOWNLOAD_FOLDERS

from pathlib import Path
import json
from datetime import datetime
from threading import Thread
import uuid

this_dir = Path(__file__).resolve().parent
history_path = this_dir / "download_history.json"

yt_bp = Blueprint(
    'yt',
    __name__,
    template_folder=str(this_dir / 'templates'),
    static_folder=str(this_dir / 'static')
)

# Logging helper function
def log_download(url, folder_path):
    from yt_dlp import YoutubeDL

    # Extract metadata using yt-dlp
    ydl_opts = {'quiet': True, 'skip_download': True}
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Title')
    except Exception as e:
        print("❌ Failed to extract title:", e)
        title = "Unknown Title"

    entry = {
        "url": url,
        "title": title,
        "folder": str(folder_path),
        "timestamp": datetime.now().isoformat()
    }

    print("📝 Logging download:", entry)
    print("📄 Writing to:", history_path)

    try:
        history = []
        if history_path.exists():
            history = json.loads(history_path.read_text())
        history.insert(0, entry)
        history_path.write_text(json.dumps(history, indent=2))
    except Exception as e:
        print("❌ Failed to log download:", e)

# Simple history logger for magnet downloads
def log_magnet_download(link, folder_path):
    entry = {
        "url": link,
        "title": "Magnet Download",
        "folder": str(folder_path),
        "timestamp": datetime.now().isoformat()
    }
    try:
        history = []
        if history_path.exists():
            history = json.loads(history_path.read_text())
        history.insert(0, entry)
        history_path.write_text(json.dumps(history, indent=2))
    except Exception as e:
        print("❌ Failed to log magnet download:", e)

# Background downloader with progress tracking
def run_download_and_track(cmd, uid):
    log_path = this_dir / f"progress_{uid}.log"
    with open(log_path, 'w') as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            f.write(line)
            f.flush()

@yt_bp.route('/yt', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url'].strip()
        quality = request.form['quality'].strip() or 'best'
        fmt = request.form['format']
        folder_key = request.form['folder']
        folder_path = DOWNLOAD_FOLDERS.get(folder_key, '/tmp')
        subtitles = 'subtitles' in request.form
        section = request.form.get('section', '').strip()

        uid = str(uuid.uuid4())[:8]
        output_template = os.path.join(folder_path, '%(title)s.%(ext)s')
        cmd = ['yt-dlp', '-f', quality, '-o', output_template, url]

        if fmt == 'audio':
            cmd += ['--extract-audio', '--audio-format', 'mp3']

        if subtitles:
            cmd += [
                '--write-auto-sub',
                '--write-sub',
                '--sub-lang', 'en',
                '--sub-format', 'srt'
            ]

        if section:
            cmd += ['--download-sections', f'*{section}']

        try:
            # Run download in background thread and track progress
            Thread(target=run_download_and_track, args=(cmd, uid), daemon=True).start()
            log_download(url, folder_path)
            flash(f"Download started for: {url} [ID: {uid}]", 'success')

        except Exception as e:
            flash(f"Error: {e}", 'danger')

        return redirect(url_for('yt.index'))

    return render_template('yt/index.html', folders=DOWNLOAD_FOLDERS)

@yt_bp.route('/yt/history')
def history():
    try:
        if history_path.exists():
            history = json.loads(history_path.read_text())
        else:
            history = []
    except Exception as e:
        print("❌ Failed to load history:", e)
        history = []

    return render_template('yt/history.html', history=history)

@yt_bp.route('/yt/clear_logs')
def clear_logs():
    deleted = 0
    for log_file in this_dir.glob("progress_*.log"):
        try:
            log_file.unlink()
            deleted += 1
        except Exception as e:
            print(f"❌ Failed to delete {log_file}: {e}")
    flash(f"🧹 Deleted {deleted} progress log files.", "success")
    return redirect(url_for('yt.history'))

@yt_bp.route('/yt/clear_history')
def clear_history():
    try:
        if history_path.exists():
            history_path.write_text("[]")
            flash("🧻 Download history cleared.", "success")
    except Exception as e:
        flash(f"Error clearing history: {e}", "danger")
    return redirect(url_for('yt.history'))

@yt_bp.route('/yt/progress/<uid>')
def get_progress(uid):
    log_path = this_dir / f"progress_{uid}.log"
    if not log_path.exists():
        return "Waiting for log...", 200
    return log_path.read_text(), 200

# ------- Magnet link downloader -------
@yt_bp.route('/yt/magnet', methods=['POST'])
def magnet_download():
    magnet = request.form.get('magnet', '').strip()
    folder_key = request.form.get('folder', '')
    folder_path = DOWNLOAD_FOLDERS.get(folder_key, '/tmp')

    if not magnet:
        flash('Magnet link required', 'danger')
        return redirect(url_for('yt.index'))

    uid = str(uuid.uuid4())[:8]
    cmd = ['aria2c', '--dir', folder_path, magnet]

    try:
        Thread(target=run_download_and_track, args=(cmd, uid), daemon=True).start()
        log_magnet_download(magnet, folder_path)
        flash(f"Magnet download started [ID: {uid}]", 'success')
    except Exception as e:
        flash(f"Magnet download error: {e}", 'danger')

    return redirect(url_for('yt.index'))
