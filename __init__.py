from flask import Blueprint, render_template, request, redirect, url_for, flash, send_from_directory, abort
import subprocess
import os
import shutil
import logging
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
logger = logging.getLogger(__name__)

PODCAST_ROOT = this_dir / "podcast_files"
PODCAST_INPUT_DIR = PODCAST_ROOT / "inputs"
PODCAST_SCRIPT_DIR = PODCAST_ROOT / "scripts"
PODCAST_AUDIO_DIR = PODCAST_ROOT / "audio"
PODCAST_JOBS_DIR = PODCAST_ROOT / "jobs"
PODCAST_PROMPTS_PATH = PODCAST_ROOT / "podcast_prompts.json"

PODCAST_ROOT.mkdir(exist_ok=True)
PODCAST_INPUT_DIR.mkdir(exist_ok=True)
PODCAST_SCRIPT_DIR.mkdir(exist_ok=True)
PODCAST_AUDIO_DIR.mkdir(exist_ok=True)
PODCAST_JOBS_DIR.mkdir(exist_ok=True)

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


def default_script_prompt_text() -> str:
    return (
        "You are a seasoned podcast writer. Turn the provided source text into a captivating "
        "podcast episode script. Requirements:\n"
        "- Start with a hook in the first 15–30 seconds.\n"
        "- Use structured segments: intro, main points, examples, recap, closing.\n"
        "- Spoken-language style with short sentences and clear transitions.\n"
        "- Optional host cues like [pause] or [music sting], but keep them practical.\n"
        "- Preserve factual content; do not fabricate. If the source is unclear, use uncertainty phrasing.\n"
        "- Tone: {tone}.\n"
    )


def default_audio_prompt_text() -> str:
    return (
        "Deliver a clear, engaging podcast narration with natural pacing and pauses. "
        "Use an articulate, warm tone, emphasize key points, and avoid rushed delivery. "
        "Keep the read conversational and easy to follow."
    )


def render_script_prompt(prompt_text: str, tone: str) -> str:
    if tone:
        if "{tone}" in prompt_text:
            return prompt_text.replace("{tone}", tone)
        return f"{prompt_text}\nTone: {tone}."
    return prompt_text


def generate_podcast_script(client: OpenAI, text: str, prompt_text: str, tone: str) -> str:
    system_prompt = render_script_prompt(prompt_text, tone)
    response = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.6,
    )
    script_text = response.choices[0].message.content or ""
    return script_text.strip()


def tts_generate_audio(client: OpenAI, script_text: str, voice: str, instructions: str) -> bytes:
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=script_text,
        instructions=instructions,
    )
    return response.read()


def load_prompts() -> dict:
    if PODCAST_PROMPTS_PATH.exists():
        try:
            return json.loads(PODCAST_PROMPTS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    prompts = {
        "script_prompts": [{"name": "Default - Script", "text": default_script_prompt_text()}],
        "audio_prompts": [{"name": "Default - Audio", "text": default_audio_prompt_text()}],
    }
    PODCAST_PROMPTS_PATH.write_text(json.dumps(prompts, indent=2))
    return prompts


def save_prompts(prompts: dict) -> None:
    PODCAST_PROMPTS_PATH.write_text(json.dumps(prompts, indent=2))


def unique_prompt_name(existing_names: set, name: str) -> str:
    if name not in existing_names:
        return name
    index = 2
    while f"{name} ({index})" in existing_names:
        index += 1
    return f"{name} ({index})"


def prompt_lookup(prompts: list[dict], name: str) -> dict | None:
    return next((prompt for prompt in prompts if prompt["name"] == name), None)


def write_job_manifest(job_id: str, manifest: dict) -> None:
    """Manifest format: {job_id, created_at, uploaded_files, script_files, audio_files, ...}."""
    job_path = PODCAST_JOBS_DIR / f"{job_id}.json"
    job_path.write_text(json.dumps(manifest, indent=2))


def load_job_manifest(job_id: str) -> dict | None:
    job_path = PODCAST_JOBS_DIR / f"{job_id}.json"
    if not job_path.exists():
        return None
    try:
        return json.loads(job_path.read_text())
    except json.JSONDecodeError:
        return None


def detect_step(manifest: dict | None, requested_step: str | None) -> int:
    if not manifest:
        return 1
    if manifest.get("audio_files"):
        return 7
    if requested_step == "audio" and manifest.get("script_files"):
        return 5
    if manifest.get("script_files"):
        return 4
    return 2


def normalize_voice(value: str | None) -> str:
    if value in VOICE_OPTIONS:
        return value
    return DEFAULT_VOICE


def normalize_tone(value: str | None) -> str:
    if value in TONE_OPTIONS:
        return value
    return TONE_OPTIONS[0]

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
    errors = []
    prompts = load_prompts()
    script_prompts = prompts.get("script_prompts", [])
    audio_prompts = prompts.get("audio_prompts", [])
    job_id = request.args.get("job_id") or request.form.get("job_id")
    manifest = load_job_manifest(job_id) if job_id else None
    requested_step = request.args.get("step")
    skip_script_checked = False

    if job_id and not manifest:
        flash("Podcast job not found. Please start again.", "danger")
        job_id = None

    # Smoke test path:
    # - upload 2 files with skip off (expect script + audio)
    # - upload 2 files with skip on (expect uploaded text as script + audio)
    # - mixed failures: one file empty should error but the other succeeds
    if request.method == 'POST':
        action = request.form.get("action")

        if action == "start":
            files = request.files.getlist("files")
            skip_script = request.form.get("skip_script") in ("on", "true", "1", "yes", "checked")
            logger.info(f"[PODCAST] skip_script={skip_script} raw={request.form.get('skip_script')}")
            logger.debug("[PODCAST] skip_script flag received during start action.")
            if not files or all(not f.filename for f in files):
                flash("Please upload at least one .txt or .md file.", "danger")
            elif len(files) > MAX_PODCAST_FILES:
                flash(f"Please upload no more than {MAX_PODCAST_FILES} files per request.", "danger")
            else:
                job_id = uuid.uuid4().hex[:12]
                manifest = {
                    "job_id": job_id,
                    "created_at": datetime.now().isoformat(),
                    "uploaded_files": [],
                    "script_files": [],
                    "audio_files": [],
                    "script_skipped": skip_script,
                    "selected_tone": TONE_OPTIONS[0],
                    "selected_voice": DEFAULT_VOICE,
                    "selected_script_prompt_name": None,
                    "selected_audio_prompt_name": None,
                }
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
                        manifest["uploaded_files"].append(
                            {
                                "original_name": safe_name,
                                "input_filename": input_name,
                                "input_path": str(input_path),
                                "base_name": base_name,
                            }
                        )
                    except Exception as e:
                        errors.append({"file": safe_name, "error": str(e)})

                if manifest["uploaded_files"]:
                    write_job_manifest(job_id, manifest)
                else:
                    flash("No valid files were uploaded.", "danger")
                    job_id = None
                    manifest = None

        elif action == "save_script_prompt":
            if not manifest:
                flash("Podcast job not found. Please start again.", "danger")
            else:
                prompt_action = request.form.get("prompt_action")
                selected_name = request.form.get("selected_script_prompt") or ""
                prompt_text = request.form.get("script_prompt_text", "").strip()
                prompt_name_input = request.form.get("prompt_name", "").strip()
                existing_names = {prompt["name"] for prompt in script_prompts}

                if prompt_action == "save_new":
                    if not prompt_name_input or not prompt_text:
                        flash("Provide a name and text to save a new prompt.", "danger")
                    else:
                        final_name = unique_prompt_name(existing_names, prompt_name_input)
                        script_prompts.append({"name": final_name, "text": prompt_text})
                        selected_name = final_name
                        flash(f"Saved new script prompt: {final_name}", "success")
                elif prompt_action == "overwrite":
                    prompt = prompt_lookup(script_prompts, selected_name)
                    if not prompt:
                        flash("Select a script prompt to overwrite.", "danger")
                    elif not prompt_text:
                        flash("Prompt text cannot be empty.", "danger")
                    else:
                        prompt["text"] = prompt_text
                        flash(f"Updated script prompt: {selected_name}", "success")
                elif prompt_action == "delete":
                    if len(script_prompts) <= 1:
                        flash("At least one script prompt is required.", "danger")
                    else:
                        script_prompts = [p for p in script_prompts if p["name"] != selected_name]
                        flash(f"Deleted script prompt: {selected_name}", "success")
                        if selected_name and manifest.get("selected_script_prompt_name") == selected_name:
                            manifest["selected_script_prompt_name"] = None
                        selected_name = ""
                else:
                    flash("Unknown prompt action.", "danger")

                prompts["script_prompts"] = script_prompts
                save_prompts(prompts)
                manifest["selected_script_prompt_name"] = selected_name or manifest.get("selected_script_prompt_name")
                write_job_manifest(manifest["job_id"], manifest)

        elif action == "generate_scripts":
            if not manifest:
                flash("Podcast job not found. Please start again.", "danger")
            else:
                skip_script = bool(manifest.get("script_skipped"))
                logger.info(f"[PODCAST] skip_script={skip_script} raw={request.form.get('skip_script')}")
                logger.debug("[PODCAST] skip_script flag applied during script generation action.")
                selected_name = request.form.get("selected_script_prompt") or ""
                prompt_text = request.form.get("script_prompt_text", "").strip()
                tone = normalize_tone(request.form.get("tone"))
                manifest["selected_tone"] = tone
                manifest["selected_script_prompt_name"] = selected_name or manifest.get("selected_script_prompt_name")
                if not skip_script:
                    if not prompt_text and selected_name:
                        prompt = prompt_lookup(script_prompts, selected_name)
                        prompt_text = prompt["text"] if prompt else ""
                    if not prompt_text:
                        flash("Script prompt text is required.", "danger")
                        prompt_text = ""
                if skip_script:
                    manifest["script_files"] = []
                    for item in manifest.get("uploaded_files", []):
                        input_path = Path(item["input_path"])
                        if not input_path.exists():
                            errors.append({"file": item["original_name"], "error": "Uploaded file missing."})
                            continue
                        text = input_path.read_text(encoding="utf-8-sig", errors="ignore")
                        if not text.strip():
                            errors.append({"file": item["original_name"], "error": "File is empty or unreadable."})
                            continue
                        try:
                            print(f"⏭️ Skipping script generation for {item['original_name']}")
                            script_text = text
                            script_name = f"{item['base_name']}_{manifest['job_id']}_podcast_script.txt"
                            script_path = PODCAST_SCRIPT_DIR / script_name
                            script_path.write_text(script_text, encoding="utf-8")
                            manifest["script_files"].append(
                                {
                                    "original_name": item["original_name"],
                                    "base_name": item["base_name"],
                                    "script_filename": script_name,
                                    "script_path": str(script_path),
                                    "script_skipped": True,
                                }
                            )
                        except Exception as e:
                            errors.append({"file": item["original_name"], "error": str(e)})
                    write_job_manifest(manifest["job_id"], manifest)
                elif prompt_text:
                    try:
                        client = get_openai_client()
                    except Exception as e:
                        flash(f"OpenAI configuration error: {e}", "danger")
                    else:
                        manifest["script_files"] = []
                        for item in manifest.get("uploaded_files", []):
                            input_path = Path(item["input_path"])
                            if not input_path.exists():
                                errors.append({"file": item["original_name"], "error": "Uploaded file missing."})
                                continue
                            text = input_path.read_text(encoding="utf-8-sig", errors="ignore")
                            if not text.strip():
                                errors.append({"file": item["original_name"], "error": "File is empty or unreadable."})
                                continue
                            try:
                                print(f"📝 Generating script for {item['original_name']}")
                                # Checked skip_script should never log Step 2.
                                script_text = generate_podcast_script(client, text, prompt_text, tone)
                                if not script_text:
                                    raise ValueError("Script generation returned empty content.")
                                script_name = f"{item['base_name']}_{manifest['job_id']}_podcast_script.txt"
                                script_path = PODCAST_SCRIPT_DIR / script_name
                                script_path.write_text(script_text, encoding="utf-8")
                                manifest["script_files"].append(
                                    {
                                        "original_name": item["original_name"],
                                        "base_name": item["base_name"],
                                        "script_filename": script_name,
                                        "script_path": str(script_path),
                                        "script_skipped": False,
                                    }
                                )
                            except Exception as e:
                                errors.append({"file": item["original_name"], "error": str(e)})
                        write_job_manifest(manifest["job_id"], manifest)

        elif action == "save_audio_prompt":
            if not manifest:
                flash("Podcast job not found. Please start again.", "danger")
            else:
                prompt_action = request.form.get("prompt_action")
                selected_name = request.form.get("selected_audio_prompt") or ""
                prompt_text = request.form.get("audio_prompt_text", "").strip()
                prompt_name_input = request.form.get("prompt_name", "").strip()
                existing_names = {prompt["name"] for prompt in audio_prompts}

                if prompt_action == "save_new":
                    if not prompt_name_input or not prompt_text:
                        flash("Provide a name and text to save a new prompt.", "danger")
                    else:
                        final_name = unique_prompt_name(existing_names, prompt_name_input)
                        audio_prompts.append({"name": final_name, "text": prompt_text})
                        selected_name = final_name
                        flash(f"Saved new audio prompt: {final_name}", "success")
                elif prompt_action == "overwrite":
                    prompt = prompt_lookup(audio_prompts, selected_name)
                    if not prompt:
                        flash("Select an audio prompt to overwrite.", "danger")
                    elif not prompt_text:
                        flash("Prompt text cannot be empty.", "danger")
                    else:
                        prompt["text"] = prompt_text
                        flash(f"Updated audio prompt: {selected_name}", "success")
                elif prompt_action == "delete":
                    if len(audio_prompts) <= 1:
                        flash("At least one audio prompt is required.", "danger")
                    else:
                        audio_prompts = [p for p in audio_prompts if p["name"] != selected_name]
                        flash(f"Deleted audio prompt: {selected_name}", "success")
                        if selected_name and manifest.get("selected_audio_prompt_name") == selected_name:
                            manifest["selected_audio_prompt_name"] = None
                        selected_name = ""
                else:
                    flash("Unknown prompt action.", "danger")

                prompts["audio_prompts"] = audio_prompts
                save_prompts(prompts)
                manifest["selected_audio_prompt_name"] = selected_name or manifest.get("selected_audio_prompt_name")
                write_job_manifest(manifest["job_id"], manifest)

        elif action == "generate_audio":
            if not manifest:
                flash("Podcast job not found. Please start again.", "danger")
            elif not manifest.get("script_files"):
                flash("No scripts found. Generate scripts first.", "danger")
            else:
                selected_name = request.form.get("selected_audio_prompt") or ""
                prompt_text = request.form.get("audio_prompt_text", "").strip()
                voice = normalize_voice(request.form.get("voice"))
                manifest["selected_voice"] = voice
                manifest["selected_audio_prompt_name"] = selected_name or manifest.get("selected_audio_prompt_name")
                if not prompt_text and selected_name:
                    prompt = prompt_lookup(audio_prompts, selected_name)
                    prompt_text = prompt["text"] if prompt else ""
                if not prompt_text:
                    flash("Audio prompt text is required.", "danger")
                else:
                    try:
                        client = get_openai_client()
                    except Exception as e:
                        flash(f"OpenAI configuration error: {e}", "danger")
                    else:
                        manifest["audio_files"] = []
                        for item in manifest.get("script_files", []):
                            script_path = Path(item["script_path"])
                            if not script_path.exists():
                                errors.append({"file": item["script_filename"], "error": "Script file missing."})
                                continue
                            script_text = script_path.read_text(encoding="utf-8-sig", errors="ignore")
                            if not script_text.strip():
                                errors.append({"file": item["script_filename"], "error": "Script file is empty."})
                                continue
                            try:
                                audio_name = f"{item['base_name']}_{manifest['job_id']}.mp3"
                                audio_path = PODCAST_AUDIO_DIR / audio_name
                                audio_bytes = tts_generate_audio(client, script_text, voice, prompt_text)
                                audio_path.write_bytes(audio_bytes)
                                manifest["audio_files"].append(
                                    {
                                        "original_name": item["original_name"],
                                        "audio_filename": audio_name,
                                        "audio_path": str(audio_path),
                                        "script_filename": item["script_filename"],
                                        "script_skipped": item.get("script_skipped", False),
                                    }
                                )
                            except Exception as e:
                                errors.append({"file": item["script_filename"], "error": str(e)})
                        write_job_manifest(manifest["job_id"], manifest)

    if manifest:
        skip_script_checked = bool(manifest.get("script_skipped"))
    elif request.method == "POST":
        skip_script_checked = request.form.get("skip_script") in ("on", "true", "1", "yes", "checked")

    return render_template(
        'yt/podcast.html',
        errors=errors,
        voice_options=VOICE_OPTIONS,
        tone_options=TONE_OPTIONS,
        job_id=job_id,
        manifest=manifest,
        current_step=detect_step(manifest, requested_step),
        script_prompts=script_prompts,
        audio_prompts=audio_prompts,
        skip_script_checked=skip_script_checked,
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
