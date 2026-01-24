from flask import Blueprint, render_template, request, redirect, url_for, flash, send_from_directory, abort
import subprocess
import os
import shutil
from config import DOWNLOAD_FOLDERS

from pathlib import Path
import json
from datetime import datetime
from threading import Thread
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename

this_dir = Path(__file__).resolve().parent
history_path = this_dir / "download_history.json"

PODCAST_ROOT = this_dir / "podcast_files"
PODCAST_INPUT_DIR = PODCAST_ROOT / "inputs"
PODCAST_SCRIPT_DIR = PODCAST_ROOT / "scripts"
PODCAST_AUDIO_DIR = PODCAST_ROOT / "audio"

PODCAST_ROOT.mkdir(exist_ok=True)
PODCAST_INPUT_DIR.mkdir(exist_ok=True)
PODCAST_SCRIPT_DIR.mkdir(exist_ok=True)
PODCAST_AUDIO_DIR.mkdir(exist_ok=True)

ENV_FILE = "openai_api_key.env"
SCRIPT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "ash"
VOICE_OPTIONS = ["alloy", "ash", "ballad", "coral", "sage", "verse"]
TONE_OPTIONS = ["informative", "energetic", "calm"]
MAX_PODCAST_FILES = 5

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

def load_api_key_from_env(env_file: str) -> str:
    """Load OPENAI_API_KEY from a .env file in the project root."""
    env_path = Path(env_file)
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")

    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY not found in {env_path}")
    return api_key


def get_openai_client() -> OpenAI:
    api_key = load_api_key_from_env(ENV_FILE)
    return OpenAI(api_key=api_key)


def is_allowed_text_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".txt", ".md"}


def build_podcast_prompt(tone: str) -> str:
    return (
        "You are a seasoned podcast writer. Turn the provided source text into a captivating "
        "podcast episode script. Requirements:\n"
        "- Start with a hook in the first 15–30 seconds.\n"
        "- Use structured segments: intro, main points, examples, recap, closing.\n"
        "- Spoken-language style with short sentences and clear transitions.\n"
        "- Optional host cues like [pause] or [music sting], but keep them practical.\n"
        "- Preserve factual content; do not fabricate. If the source is unclear, use uncertainty phrasing.\n"
        f"- Tone: {tone}.\n"
    )


def generate_podcast_script(client: OpenAI, text: str, tone: str) -> str:
    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": build_podcast_prompt(tone)},
            {"role": "user", "content": text},
        ],
        temperature=0.6,
    )
    script_text = response.choices[0].message.content or ""
    return script_text.strip()


def tts_generate_audio(client: OpenAI, script_text: str, voice: str) -> bytes:
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=script_text,
        instructions=(
            "Speak clearly with natural pauses and an engaging, podcast-ready delivery."
        ),
    )
    return response.read()

# Background downloader with progress tracking
def run_download_and_track(cmd, uid):
    """Execute a shell command and stream output to a progress log."""

    log_path = this_dir / f"progress_{uid}.log"
    with open(log_path, "w") as f:
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except FileNotFoundError as e:
            # Write the error message so the user can see what went wrong
            f.write(str(e))
            return

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
        audio_container = request.form.get('audio_container', 'mp3').strip() or 'mp3'
        video_container = request.form.get('video_container', 'mp4').strip() or 'mp4'

        uid = str(uuid.uuid4())[:8]
        output_template = os.path.join(folder_path, '%(title)s.%(ext)s')
        cmd = ['yt-dlp', '-f', quality, '-o', output_template, url]

        if fmt == 'audio':
            cmd += ['--extract-audio', '--audio-format', audio_container]
        else:
            cmd += ['--remux-video', video_container]

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

@yt_bp.route('/yt/podcast', methods=['GET', 'POST'])
def podcast():
    results = []
    errors = []
    selected_voice = request.form.get("voice", DEFAULT_VOICE)
    selected_tone = request.form.get("tone", TONE_OPTIONS[0])

    if request.method == 'POST':
        files = request.files.getlist("files")
        if not files or all(not f.filename for f in files):
            flash("Please upload at least one .txt or .md file.", "danger")
            return render_template(
                'yt/podcast.html',
                results=results,
                errors=errors,
                voice_options=VOICE_OPTIONS,
                tone_options=TONE_OPTIONS,
                selected_voice=selected_voice,
                selected_tone=selected_tone,
            )

        if len(files) > MAX_PODCAST_FILES:
            flash(f"Please upload no more than {MAX_PODCAST_FILES} files per request.", "danger")
            return render_template(
                'yt/podcast.html',
                results=results,
                errors=errors,
                voice_options=VOICE_OPTIONS,
                tone_options=TONE_OPTIONS,
                selected_voice=selected_voice,
                selected_tone=selected_tone,
            )

        try:
            client = get_openai_client()
        except Exception as e:
            flash(f"OpenAI configuration error: {e}", "danger")
            return render_template(
                'yt/podcast.html',
                results=results,
                errors=errors,
                voice_options=VOICE_OPTIONS,
                tone_options=TONE_OPTIONS,
                selected_voice=selected_voice,
                selected_tone=selected_tone,
            )

        for upload in files:
            original_name = upload.filename or ""
            safe_name = secure_filename(original_name)
            if not safe_name:
                errors.append({"file": original_name, "error": "Invalid filename."})
                continue

            if not is_allowed_text_file(safe_name):
                errors.append({"file": safe_name, "error": "Only .txt and .md files are allowed."})
                continue

            unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            base_name = Path(safe_name).stem
            suffix = Path(safe_name).suffix.lower()

            input_name = f"{base_name}_{unique_id}{suffix}"
            input_path = PODCAST_INPUT_DIR / input_name

            try:
                upload.save(input_path)
                print(f"📥 Saved upload: {input_path}")

                text = input_path.read_text(encoding="utf-8-sig", errors="ignore")
                if not text.strip():
                    raise ValueError("File is empty or could not be read.")

                script_text = generate_podcast_script(client, text, selected_tone)
                if not script_text:
                    raise ValueError("Script generation returned empty content.")

                script_name = f"{base_name}_{unique_id}_podcast_script.txt"
                script_path = PODCAST_SCRIPT_DIR / script_name
                script_path.write_text(script_text, encoding="utf-8")
                print(f"📝 Script generated: {script_path}")

                audio_name = f"{base_name}_{unique_id}.mp3"
                audio_path = PODCAST_AUDIO_DIR / audio_name
                audio_bytes = tts_generate_audio(client, script_text, selected_voice)
                audio_path.write_bytes(audio_bytes)
                print(f"🔊 Audio generated: {audio_path}")

                results.append({
                    "original_name": safe_name,
                    "script_name": script_name,
                    "audio_name": audio_name,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"❌ Podcast processing failed for {safe_name}: {e}")
                errors.append({"file": safe_name, "error": str(e)})

    return render_template(
        'yt/podcast.html',
        results=results,
        errors=errors,
        voice_options=VOICE_OPTIONS,
        tone_options=TONE_OPTIONS,
        selected_voice=selected_voice,
        selected_tone=selected_tone,
    )


@yt_bp.route('/yt/podcast/download/<category>/<path:filename>')
def podcast_download(category, filename):
    directories = {
        "scripts": PODCAST_SCRIPT_DIR,
        "audio": PODCAST_AUDIO_DIR,
        "inputs": PODCAST_INPUT_DIR,
    }
    if category not in directories:
        abort(404)
    safe_name = secure_filename(filename)
    if safe_name != filename:
        abort(404)
    return send_from_directory(directories[category], filename, as_attachment=True)


@yt_bp.route('/yt/podcast/media/<path:filename>')
def podcast_media(filename):
    safe_name = secure_filename(filename)
    if safe_name != filename:
        abort(404)
    return send_from_directory(PODCAST_AUDIO_DIR, filename, as_attachment=False)

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

    # Verify that aria2c is available before attempting the download
    if shutil.which('aria2c') is None:
        flash('aria2c is required for magnet downloads. Please install it first.', 'danger')
        return redirect(url_for('yt.index'))

    cmd = ['aria2c', '--dir', folder_path, magnet]

    try:
        Thread(target=run_download_and_track, args=(cmd, uid), daemon=True).start()
        log_magnet_download(magnet, folder_path)
        flash(f"Magnet download started [ID: {uid}]", 'success')
    except Exception as e:
        flash(f"Magnet download error: {e}", 'danger')

    return redirect(url_for('yt.index'))
