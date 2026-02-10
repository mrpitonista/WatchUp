from flask import Blueprint, render_template, request, redirect, url_for, flash, send_from_directory, abort
import subprocess
import os
import shutil
import logging
import re
import sys
from config import DOWNLOAD_FOLDERS

from pathlib import Path
import json
from datetime import datetime
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from clipper_utils import (
    SubtitleCue,
    build_blocks,
    blocks_to_prompt_text,
    detect_language_text,
    find_matching_video,
    generate_clip_subtitles,
    get_google_translate_client,
    list_clipper_jobs,
    ms_to_timestamp,
    mux_mkv_ffmpeg,
    parse_srt,
    parse_timestamp_to_ms,
    parse_vtt,
    safe_list_media_files,
    translate_texts,
    write_srt,
    write_vtt,
)

this_dir = Path(__file__).resolve().parent
history_path = this_dir / "download_history.json"
logger = logging.getLogger(__name__)

PODCAST_ROOT = this_dir / "podcast_files"
PODCAST_UPLOAD_DIR = PODCAST_ROOT / "uploads"
PODCAST_SCRIPT_DIR = PODCAST_ROOT / "scripts"
PODCAST_AUDIO_DIR = PODCAST_ROOT / "audio"
PODCAST_JOBS_DIR = PODCAST_ROOT / "jobs"
PODCAST_OUTPUT_DIR = PODCAST_ROOT / "output"
PODCAST_PROMPTS_PATH = PODCAST_ROOT / "podcast_prompts.json"
SMB_BASE_URL = os.getenv(
    "WATCHUP_SMB_BASE_URL",
    "smb://MacMiniMedia._smb._tcp.local/Admin/YT_Dashboard/podcast_files",
)

MEDIA_OTHER_DIR = this_dir.parent / "Media" / "Other"
MEDIA_CLIPS_DIR = this_dir.parent / "Media" / "Clips"
CLIPPER_ROOT = this_dir / "clipper_files"
CLIPPER_SUMMARY_DIR = CLIPPER_ROOT / "summaries"
CLIPPER_BLOCKS_DIR = CLIPPER_ROOT / "blocks"
CLIPPER_JOBS_DIR = CLIPPER_ROOT / "jobs"
CLIPPER_PREVIOUS_JOBS_LIMIT = 15
CLIPPER_TRANSLATE_CREDS_PATH = this_dir / "Google_translate_api-463712-b435ed9c8fa8.json"

PODCAST_ROOT.mkdir(exist_ok=True)
PODCAST_UPLOAD_DIR.mkdir(exist_ok=True)
PODCAST_SCRIPT_DIR.mkdir(exist_ok=True)
PODCAST_AUDIO_DIR.mkdir(exist_ok=True)
PODCAST_JOBS_DIR.mkdir(exist_ok=True)
PODCAST_OUTPUT_DIR.mkdir(exist_ok=True)

CLIPPER_ROOT.mkdir(exist_ok=True)
CLIPPER_SUMMARY_DIR.mkdir(exist_ok=True)
CLIPPER_BLOCKS_DIR.mkdir(exist_ok=True)
CLIPPER_JOBS_DIR.mkdir(exist_ok=True)
MEDIA_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

ENV_FILE = "openai_api_key.env"
SCRIPT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "ash"
VOICE_OPTIONS = ["alloy", "ash", "ballad", "coral", "sage", "verse"]
TONE_OPTIONS = ["informative", "energetic", "calm"]
MAX_PODCAST_FILES = 5
SCRIPT_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=2)
SCRIPT_CHUNK_TIMEOUT = 90
SCRIPT_MAX_RETRIES = 1

CLIPPER_SYSTEM_PROMPT = (
    "You are a precise video timeline analyst. You will be given subtitle blocks in the form:\n"
    "[start–end] text...\n"
    "Your job is to extract the most important moments and return a strict JSON object that can "
    "drive automated video clipping. Do not invent timestamps. Use only timestamps that fall "
    "within the provided blocks. Keep segments non-overlapping and in chronological order."
)

CLIPPER_USER_PROMPT_TEMPLATE = (
    "TASK\n"
    "From the subtitle blocks below, produce a timestamped summary suitable for automated clip generation.\n\n"
    "OUTPUT FORMAT (STRICT JSON ONLY)\n"
    "Return exactly one JSON object with this schema:\n\n"
    "{\n"
    "  \"title\": string,\n"
    "  \"language\": string,\n"
    "  \"segments\": [\n"
    "    {\n"
    "      \"start\": \"HH:MM:SS.mmm\",\n"
    "      \"end\": \"HH:MM:SS.mmm\",\n"
    "      \"headline\": string,\n"
    "      \"summary\": string,\n"
    "      \"why_it_matters\": string,\n"
    "      \"keywords\": [string, ...]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "RULES (NON-NEGOTIABLE)\n"
    "1) Choose 8–12 segments total.\n"
    "2) Segments must be chronological and must NOT overlap (end <= next start).\n"
    "3) Each segment must be between 20 seconds and 3 minutes.\n"
    "4) Start and end times must be within the provided timeline. Do NOT use timestamps outside it.\n"
    "5) Prefer boundaries that align with block boundaries you see.\n"
    "6) Avoid duplication: each segment must cover a different idea/event.\n"
    "7) Strict JSON only: no markdown, no commentary.\n"
    "8) Use the exact key \"why_it_matters\" (underscore). Do NOT use dots or other variants.\n\n"
    "QUALITY GOAL\n"
    "Select moments that capture:\n"
    "- the core argument / story arc\n"
    "- key claims or decisions\n"
    "- notable examples/demos\n"
    "- conclusions / takeaways\n\n"
    "SUBTITLE BLOCKS (SOURCE OF TRUTH)\n"
    "{blocks_text}"
)

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


def resolve_job_folder(job_id: str, kind: str) -> Path | None:
    folders = {
        "uploads": PODCAST_UPLOAD_DIR,
        "scripts": PODCAST_SCRIPT_DIR,
        "audio": PODCAST_AUDIO_DIR,
    }
    base_dir = folders.get(kind)
    if not base_dir:
        return None
    job_dir = (base_dir / job_id).resolve()
    try:
        job_dir.relative_to(base_dir.resolve())
    except ValueError:
        return None
    return job_dir


def get_openai_client(max_retries: int | None = None) -> OpenAI:
    api_key = load_api_key_from_env(ENV_FILE)
    if max_retries is None:
        return OpenAI(api_key=api_key)
    return OpenAI(api_key=api_key, max_retries=max_retries)


def clipper_job_path(job_id: str) -> Path:
    return CLIPPER_JOBS_DIR / f"{job_id}.json"


def load_clipper_job(job_id: str) -> dict | None:
    path = clipper_job_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def write_clipper_job(job_id: str, payload: dict) -> None:
    clipper_job_path(job_id).write_text(json.dumps(payload, indent=2))


def resolve_media_other_file(filename: str) -> Path | None:
    if not filename or Path(filename).name != filename:
        return None
    candidate = (MEDIA_OTHER_DIR / filename).resolve()
    try:
        candidate.relative_to(MEDIA_OTHER_DIR.resolve())
    except ValueError:
        return None
    if not candidate.exists():
        return None
    return candidate


def resolve_clips_file(filename: str) -> Path | None:
    if not filename or Path(filename).name != filename:
        return None
    candidate = (MEDIA_CLIPS_DIR / filename).resolve()
    try:
        candidate.relative_to(MEDIA_CLIPS_DIR.resolve())
    except ValueError:
        return None
    if not candidate.exists():
        return None
    return candidate


def format_clip_timestamp(value: str) -> str:
    return value.replace(":", "-").replace(".", "_")


def format_duration_seconds(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def build_translation_detection_sample(cues: list[SubtitleCue], max_cues: int = 30, max_chars: int = 4000) -> str:
    texts = [cue.text.strip() for cue in cues if cue.text and cue.text.strip()]
    sample = " ".join(texts[:max_cues])
    return sample[:max_chars]


def build_translated_cues(cues: list[SubtitleCue], translated_texts: list[str]) -> list[SubtitleCue]:
    translated_cues: list[SubtitleCue] = []
    for cue, translated_text in zip(cues, translated_texts):
        translated_cues.append(
            SubtitleCue(
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=translated_text.strip() if translated_text else "",
            )
        )
    return translated_cues


def maybe_translate_clip_subtitles(
    *,
    job_id: str,
    segment_index: int,
    subtitle_out_path: Path,
    subtitle_vtt_out_path: Path | None,
    translate_target: str,
) -> tuple[str | None, str | None]:
    if translate_target not in {"en", "it"}:
        return None, None

    logger.info(
        "[CLIPPER] translate requested target=%s job_id=%s seg=%s",
        translate_target,
        job_id,
        segment_index + 1,
    )

    if not CLIPPER_TRANSLATE_CREDS_PATH.exists():
        logger.error("[CLIPPER] translate credentials file missing: %s", CLIPPER_TRANSLATE_CREDS_PATH)
        flash("Google Translate credentials file is missing. Skipping subtitle translation.", "warning")
        return None, None

    try:
        cues = parse_srt(subtitle_out_path)
    except Exception as exc:
        logger.error("[CLIPPER] translate parse failed path=%s error=%s", subtitle_out_path, exc)
        return None, None

    if not cues:
        return None, None

    sample_text = build_translation_detection_sample(cues)
    if not sample_text:
        return None, None

    try:
        translate_client = get_google_translate_client(CLIPPER_TRANSLATE_CREDS_PATH)
        detected_language = detect_language_text(translate_client, sample_text)
        logger.info("[CLIPPER] detected subtitle language=%s", detected_language or "unknown")
        if detected_language in {"en", "it"}:
            logger.info("[CLIPPER] translate skipped (detected in {en,it})")
            return None, None

        source_texts = [cue.text if cue.text.strip() else "" for cue in cues]
        translated_texts = translate_texts(translate_client, source_texts, translate_target)
        translated_cues = build_translated_cues(cues, translated_texts)

        translated_srt_path = subtitle_out_path.with_suffix(f".{translate_target}.srt")
        write_srt(translated_cues, translated_srt_path)
        translated_vtt_filename: str | None = None
        if subtitle_vtt_out_path and subtitle_vtt_out_path.exists():
            translated_vtt_path = subtitle_vtt_out_path.with_suffix(f".{translate_target}.vtt")
            write_vtt(translated_cues, translated_vtt_path)
            translated_vtt_filename = translated_vtt_path.name
        logger.info("[CLIPPER] translated cues=%s -> %s", len(translated_cues), translated_srt_path)
        return translated_srt_path.name, translated_vtt_filename
    except Exception as exc:
        logger.error("[CLIPPER] translate failed target=%s error=%s", translate_target, exc)
        return None, None


def job_status_path(job_id: str) -> Path:
    return PODCAST_OUTPUT_DIR / job_id / "status.json"


def load_job_status(job_id: str) -> dict | None:
    status_path = job_status_path(job_id)
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text())
    except json.JSONDecodeError:
        return None


def write_job_status(job_id: str, status: dict) -> None:
    status_path = job_status_path(job_id)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2))


def update_job_status(job_id: str, **updates: object) -> dict:
    current = load_job_status(job_id) or {
        "job_id": job_id,
        "state": "queued",
        "current_file": None,
        "current_chunk": 0,
        "total_chunks": 0,
        "last_message": "",
        "errors": [],
        "outputs": {},
    }
    current.update(updates)
    write_job_status(job_id, current)
    return current


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


def generate_podcast_script(
    client: OpenAI,
    text: str,
    prompt_text: str,
    tone: str,
    model: str | None = None,
) -> str:
    system_prompt = render_script_prompt(prompt_text, tone)
    script_model = model or SCRIPT_MODEL
    allowed_models = {"gpt-4o-mini", "gpt-5-mini"}
    if script_model not in allowed_models:
        raise ValueError(f"Unsupported script model: {script_model}")
    kwargs = {
        "model": script_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    }
    if script_model == "gpt-4o-mini":
        kwargs["temperature"] = 0.6
    logger.info(
        f"[SCRIPT] model={script_model} has_temperature={'temperature' in kwargs} "
        f"temperature={kwargs.get('temperature')} has_top_p={'top_p' in kwargs}"
    )
    response = client.chat.completions.create(**kwargs)
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
    if manifest.get("script_generation_state") in {"queued", "running", "error"}:
        return 3
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


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(re.sub(r"[ \t]+", " ", line.strip()) for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def detect_heading_structure(text: str) -> bool:
    return any(re.match(r"^(# |## )", line) for line in text.split("\n"))


def split_text_with_metadata(text: str, max_chars: int = 7000) -> tuple[list[str], str, bool]:
    headings_used = detect_heading_structure(text)
    strategy_used = set()

    def hard_cut(value: str, limit: int) -> list[str]:
        return [value[i : i + limit] for i in range(0, len(value), limit)]

    def pack_units(units: list[str], limit: int, sep: str = "\n\n") -> list[str]:
        if not units:
            return []
        packed = []
        current = ""
        for unit in units:
            if not unit:
                continue
            candidate = unit if not current else f"{current}{sep}{unit}"
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    packed.append(current)
                current = unit
        if current:
            packed.append(current)
        return packed

    def split_paragraph(paragraph: str, limit: int) -> list[str]:
        if len(paragraph) <= limit:
            return [paragraph]
        strategy_used.add("sentences")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
        if not sentences:
            strategy_used.add("hardcut")
            return hard_cut(paragraph, limit)
        chunks = []
        current = ""
        for sentence in sentences:
            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(sentence) > limit:
                    strategy_used.add("hardcut")
                    chunks.extend(hard_cut(sentence, limit))
                    current = ""
                else:
                    current = sentence
        if current:
            chunks.append(current)
        return chunks

    def split_body_into_chunks(body: str, limit: int) -> list[str]:
        if not body:
            return []
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        chunks = []
        for paragraph in paragraphs:
            chunks.extend(split_paragraph(paragraph, limit))
        return chunks

    def extract_heading_prefix(unit_text: str) -> tuple[str, str]:
        lines = unit_text.split("\n")
        heading_lines = []
        for line in lines:
            if line.startswith("# ") or line.startswith("## "):
                heading_lines.append(line)
            else:
                break
        if not heading_lines:
            return "", unit_text
        body = "\n".join(lines[len(heading_lines) :]).lstrip()
        return "\n".join(heading_lines), body

    def combine_with_prefix(body_chunks: list[str], prefix: str, limit: int) -> list[str]:
        if not body_chunks:
            return [prefix] if prefix else []
        continuation = "(continued)" if prefix else ""
        combined = []
        for index, chunk in enumerate(body_chunks):
            if index == 0 and prefix:
                combined.append(f"{prefix}\n\n{chunk}".strip())
            elif continuation:
                combined.append(f"{continuation}\n\n{chunk}".strip())
            else:
                combined.append(chunk.strip())
        return combined

    def split_unit_with_fallback(unit_text: str) -> list[str]:
        if len(unit_text) <= max_chars:
            return [unit_text.strip()]
        prefix, body = extract_heading_prefix(unit_text)
        available = max_chars
        if prefix:
            first_limit = max_chars - len(prefix) - 2
            continuation_limit = max_chars - len("(continued)") - 2
            available = max(1, min(first_limit, continuation_limit))
        body_chunks = split_body_into_chunks(body, available)
        if not body_chunks and body:
            strategy_used.add("hardcut")
            body_chunks = hard_cut(body, available)
        return combine_with_prefix(body_chunks or [""] if prefix else body_chunks, prefix, available)

    def log_chunking(chunks: list[str], strategy: str, headings: bool) -> None:
        logger.info(
            "[SEGMENTER] headings=%s max_chars=%s text_len=%s chunks=%s strategy=%s",
            headings,
            max_chars,
            len(text),
            len(chunks),
            strategy,
        )
        if len(chunks) > max(10, (len(text) // max_chars) + 5):
            logger.warning("[SEGMENTER] Excessive chunk count detected...")

    if len(text) <= max_chars:
        strategy = "headings" if headings_used else "paragraphs"
        chunks = [text.strip()]
        log_chunking(chunks, strategy, headings_used)
        return chunks, strategy, headings_used

    if headings_used:
        units = []
        current_chapter = None
        current_lines = []
        for line in text.split("\n"):
            if line.startswith("# "):
                if current_lines:
                    units.append("\n".join(current_lines).strip())
                current_chapter = line.strip()
                current_lines = [current_chapter]
            elif line.startswith("## "):
                if current_lines and current_lines != [current_chapter]:
                    units.append("\n".join(current_lines).strip())
                    current_lines = []
                if current_chapter:
                    current_lines = [current_chapter, line.strip()]
                else:
                    current_lines = [line.strip()]
            else:
                if current_lines:
                    current_lines.append(line)
                else:
                    current_lines = [line]
        if current_lines:
            units.append("\n".join(current_lines).strip())
        packed = []
        current = ""
        for unit in units:
            unit_chunks = split_unit_with_fallback(unit)
            for unit_chunk in unit_chunks:
                candidate = unit_chunk if not current else f"{current}\n\n{unit_chunk}"
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        packed.append(current.strip())
                    current = unit_chunk
        if current:
            packed.append(current.strip())
        strategy = "hardcut" if "hardcut" in strategy_used else "sentences" if "sentences" in strategy_used else "headings"
        log_chunking(packed, strategy, headings_used)
        return packed, strategy, headings_used

    strategy_used.add("paragraphs")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units = []
    for paragraph in paragraphs:
        units.extend(split_paragraph(paragraph, max_chars))
    paragraph_chunks = pack_units(units, max_chars, sep="\n\n")
    strategy = "hardcut" if "hardcut" in strategy_used else "sentences" if "sentences" in strategy_used else "paragraphs"
    log_chunking(paragraph_chunks, strategy, headings_used)
    return paragraph_chunks, strategy, headings_used


def split_for_tts(text: str, max_chars: int = 7000) -> list[str]:
    chunks, _strategy, _headings_used = split_text_with_metadata(text, max_chars=max_chars)
    return chunks


def clamp_max_chars(value: int, minimum: int = 3000, maximum: int = 12000) -> int:
    return max(minimum, min(maximum, value))


def merge_mp3_parts_ffmpeg(part_paths: list[Path], out_path: Path) -> tuple[bool, str]:
    if not part_paths:
        return False, "No audio parts to merge."
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg is required to merge audio parts."
    concat_list_path = out_path.parent / f"{out_path.stem}_concat.txt"
    try:
        concat_list_path.write_text(
            "\n".join(f"file '{path.as_posix()}'" for path in part_paths),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error("[PODCAST] ffmpeg merge failed: %s", result.stderr.strip())
            return False, result.stderr.strip() or "ffmpeg merge failed."
        logger.info("[PODCAST] ffmpeg merge OK -> %s", out_path)
        return True, ""
    finally:
        if concat_list_path.exists():
            concat_list_path.unlink()


def log_chunk_warning(job_id: str, file_name: str, chunk_count: int, max_chars: int) -> None:
    if chunk_count > 80:
        logger.warning(
            "[PODCAST] job_id=%s file=%s chunks=%s. Consider increasing max_chars (current=%s).",
            job_id,
            file_name,
            chunk_count,
            max_chars,
        )


def write_chunk_error_file(job_script_dir: Path, base_name: str, index: int, error: str) -> Path:
    error_name = f"{base_name}_script_part_{index:02d}.error.txt"
    error_path = job_script_dir / error_name
    error_path.write_text(error, encoding="utf-8")
    return error_path


def process_script_job(
    job_id: str,
    prompt_text: str,
    tone: str,
    script_model: str,
    skip_script: bool,
) -> None:
    manifest = load_job_manifest(job_id)
    if not manifest:
        update_job_status(job_id, state="error", last_message="Job manifest missing.")
        return
    try:
        update_job_status(
            job_id,
            state="running",
            current_file=None,
            current_chunk=0,
            total_chunks=0,
            last_message="Script generation started.",
            errors=[],
        )
        logger.info("[PODCAST] job_id=%s script_model=%s skip_script=%s", job_id, script_model, skip_script)
        auto_split = bool(manifest.get("auto_split", True))
        max_chars = clamp_max_chars(int(manifest.get("max_chars", 7000)))
        client = None
        if not skip_script:
            client = get_openai_client(max_retries=SCRIPT_MAX_RETRIES)
        script_files = []
        errors = []
        for item in manifest.get("uploaded_files", []):
            input_path = Path(item["input_path"])
            job_script_dir = PODCAST_SCRIPT_DIR / job_id
            job_script_dir.mkdir(parents=True, exist_ok=True)
            if not input_path.exists():
                error = {"file": item["original_name"], "error": "Uploaded file missing."}
                errors.append(error)
                update_job_status(job_id, errors=errors, last_message=error["error"])
                continue
            text = input_path.read_text(encoding="utf-8-sig", errors="ignore")
            normalized_text = normalize_text(text)
            if not normalized_text.strip():
                error = {"file": item["original_name"], "error": "File is empty or unreadable."}
                errors.append(error)
                update_job_status(job_id, errors=errors, last_message=error["error"])
                continue
            chunks = [normalized_text]
            strategy = "none"
            headings_detected = detect_heading_structure(normalized_text)
            headings_used = False
            logger.info(
                "[PODCAST] job_id=%s file=%s auto_split=%s max_chars=%s headings=%s chunks=%s skip_script=%s script_model=%s",
                job_id,
                item["original_name"],
                auto_split,
                max_chars,
                headings_detected,
                len(chunks),
                skip_script,
                script_model,
            )
            update_job_status(
                job_id,
                current_file=item["original_name"],
                current_chunk=0,
                total_chunks=1,
                last_message=f"Script generation: 0/1 for {item['original_name']}.",
            )
            script_parts = []
            if skip_script:
                script_text = normalized_text
                full_name = f"{item['base_name']}_full_script_source.txt"
                logger.info("[SCRIPT] single-pass model=source chars=%s", len(script_text))
            else:
                start_time = datetime.now()
                logger.info("[SCRIPT] single-pass model=%s chars=%s", script_model, len(normalized_text))
                try:
                    chunk_client = client.with_options(timeout=SCRIPT_CHUNK_TIMEOUT, max_retries=SCRIPT_MAX_RETRIES)
                    script_text = generate_podcast_script(
                        chunk_client,
                        normalized_text,
                        prompt_text,
                        tone,
                        model=script_model,
                    )
                    if not script_text:
                        raise ValueError("Script generation returned empty content.")
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    error_message = f"Script generation failed after {duration:.2f}s: {e}"
                    logger.error(
                        "[PODCAST] job_id=%s file=%s ERROR %s",
                        job_id,
                        item["original_name"],
                        error_message,
                    )
                    errors.append({"file": item["original_name"], "error": error_message})
                    update_job_status(job_id, errors=errors, last_message=error_message)
                    continue
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    "[PODCAST] job_id=%s file=%s single-pass END %.2fs",
                    job_id,
                    item["original_name"],
                    duration,
                )
                full_name = f"{item['base_name']}_full_script_{script_model}.txt"
            full_path = job_script_dir / full_name
            full_path.write_text(script_text, encoding="utf-8")
            update_job_status(
                job_id,
                current_chunk=1,
                last_message=f"Script generation: 1/1 for {item['original_name']}.",
            )
            script_files.append(
                {
                    "original_name": item["original_name"],
                    "base_name": item["base_name"],
                    "script_parts": script_parts,
                    "script_full_filename": f"{job_id}/{full_name}",
                    "script_full_path": str(full_path),
                    "script_skipped": skip_script,
                    "chunk_count": 1,
                    "auto_split": auto_split,
                    "max_chars": max_chars,
                    "headings_detected": headings_detected,
                    "headings_used": headings_used,
                    "chunk_strategy": strategy,
                    "script_model": script_model,
                }
            )
        manifest["script_files"] = script_files
        manifest["script_generation_state"] = "done"
        write_job_manifest(job_id, manifest)
        update_job_status(
            job_id,
            state="done",
            current_file=None,
            current_chunk=0,
            total_chunks=0,
            last_message="Script generation complete.",
            errors=errors,
            outputs={"script_files": script_files},
        )
    except Exception as e:
        logger.exception("[PODCAST] job_id=%s script job failed.", job_id)
        manifest["script_generation_state"] = "error"
        write_job_manifest(job_id, manifest)
        update_job_status(job_id, state="error", last_message=str(e))

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
        quality = 'bv*+ba/b'
        fmt = request.form['format']
        folder_key = request.form['folder']
        folder_path = DOWNLOAD_FOLDERS.get(folder_key, '/tmp')
        subtitles = 'subtitles' in request.form
        section = request.form.get('section', '').strip()
        audio_container = request.form.get('audio_container', 'mp3').strip() or 'mp3'
        video_container = request.form.get('video_container', 'mp4').strip() or 'mp4'

        uid = str(uuid.uuid4())[:8]
        output_template = os.path.join(folder_path, '%(title)s.%(ext)s')
        cmd = [
            'yt-dlp',
            '--cookies-from-browser', 'chrome',
            '--extractor-args', 'youtube:player_client=android',
            '-f', quality,
            '-S', 'proto:https,res,ext:mp4:m4a',
            '-o', output_template,
            url
        ]

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

    return render_template('yt/index.html', folders=DOWNLOAD_FOLDERS, active_page="yt")


@yt_bp.route('/yt/home')
def home():
    return render_template('yt/landing.html', active_page="home")


@yt_bp.route('/yt/clipper', methods=['GET'])
def clipper():
    subtitle_files = safe_list_media_files(MEDIA_OTHER_DIR, [".srt", ".vtt"])
    video_files = safe_list_media_files(MEDIA_OTHER_DIR, [".mp4", ".mkv", ".webm", ".mov"])
    previous_jobs = list_clipper_jobs(
        jobs_dir=CLIPPER_JOBS_DIR,
        summary_dir=CLIPPER_SUMMARY_DIR,
        limit=CLIPPER_PREVIOUS_JOBS_LIMIT,
    )
    return render_template(
        'yt/clipper.html',
        active_page="clipper",
        subtitle_files=subtitle_files,
        video_files=video_files,
        previous_jobs=previous_jobs,
    )


@yt_bp.route('/yt/clipper/analyze', methods=['POST'])
def clipper_analyze():
    subtitle_filename = (request.form.get("subtitle_file") or "").strip()
    video_filename = (request.form.get("video_file") or "").strip()
    summary_model = (request.form.get("summary_model") or "gpt-4o-mini").strip()
    allowed_summary_models = {"gpt-4o-mini", "gpt-5-mini", "gpt-5.2"}
    if summary_model not in allowed_summary_models:
        summary_model = "gpt-4o-mini"
    subtitle_path = resolve_media_other_file(subtitle_filename)
    if not subtitle_path:
        flash("Please select a valid subtitle file from Admin/Media/Other.", "danger")
        return redirect(url_for('yt.clipper'))

    if subtitle_path.suffix.lower() == ".srt":
        cues = parse_srt(subtitle_path)
    elif subtitle_path.suffix.lower() == ".vtt":
        cues = parse_vtt(subtitle_path)
    else:
        flash("Unsupported subtitle format. Use .srt or .vtt.", "danger")
        return redirect(url_for('yt.clipper'))

    if not cues:
        flash("No subtitle cues were found in the selected file.", "danger")
        return redirect(url_for('yt.clipper'))

    if video_filename:
        video_path = resolve_media_other_file(video_filename)
        if not video_path:
            flash("Selected video file is not available in Admin/Media/Other.", "danger")
            return redirect(url_for('yt.clipper'))
    else:
        match = find_matching_video(subtitle_path)
        video_path = resolve_media_other_file(match) if match else None

    if not video_path:
        flash("No matching video file found. Please select a video manually.", "danger")
        return redirect(url_for('yt.clipper'))

    job_id = uuid.uuid4().hex[:12]
    blocks = build_blocks(cues, target_seconds=45, max_chars=1400)
    if not blocks:
        flash("Failed to build subtitle blocks for analysis.", "danger")
        return redirect(url_for('yt.clipper'))

    job_blocks_dir = CLIPPER_BLOCKS_DIR / job_id
    job_summary_dir = CLIPPER_SUMMARY_DIR / job_id
    job_blocks_dir.mkdir(parents=True, exist_ok=True)
    job_summary_dir.mkdir(parents=True, exist_ok=True)

    blocks_preview = blocks_to_prompt_text(blocks)
    blocks_preview_path = job_blocks_dir / f"{subtitle_path.stem}_blocks_preview.txt"
    blocks_json_path = job_blocks_dir / f"{subtitle_path.stem}_blocks.json"

    blocks_preview_path.write_text(blocks_preview)
    blocks_payload = [
        {
            "start": ms_to_timestamp(block["start_ms"]),
            "end": ms_to_timestamp(block["end_ms"]),
            "text": block["text"],
            "start_ms": block["start_ms"],
            "end_ms": block["end_ms"],
        }
        for block in blocks
    ]
    blocks_json_path.write_text(json.dumps(blocks_payload, indent=2))

    logger.info(
        "[CLIPPER] job_id=%s subtitle=%s video=%s blocks=%s",
        job_id,
        subtitle_path.name,
        video_path.name,
        len(blocks),
    )

    prompt_text = CLIPPER_USER_PROMPT_TEMPLATE.replace("{blocks_text}", blocks_preview)
    logger.info("[CLIPPER] openai request chars=%s", len(prompt_text))
    client = get_openai_client()
    summary_data: dict
    try:
        response = client.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": CLIPPER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
        )
        raw_summary = response.choices[0].message.content or ""
    except Exception as exc:
        logger.exception("[CLIPPER] openai request failed")
        flash(f"OpenAI request failed: {exc}", "danger")
        return redirect(url_for('yt.clipper'))

    try:
        summary_data = json.loads(raw_summary)
        if not isinstance(summary_data, dict):
            raise ValueError("Summary JSON is not an object.")
    except (json.JSONDecodeError, ValueError):
        summary_data = {"error": "Invalid JSON from OpenAI", "raw": raw_summary, "segments": []}
        flash("OpenAI returned invalid JSON. Check the summary output for details.", "danger")
    else:
        segments = summary_data.get("segments", [])
        if not isinstance(segments, list):
            segments = []
        normalized_segments = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            if "why_it.matters" in segment:
                segment["why_it_matters"] = segment.pop("why_it.matters")
            if segment.get("why_it_matters") is None or "why_it_matters" not in segment:
                segment["why_it_matters"] = ""
            normalized_segments.append(segment)
        summary_data["segments"] = normalized_segments

    summary_path = job_summary_dir / f"{subtitle_path.stem}_summary.json"
    summary_pretty_path = job_summary_dir / f"{subtitle_path.stem}_summary_pretty.json"
    summary_path.write_text(json.dumps(summary_data))
    summary_pretty_path.write_text(json.dumps(summary_data, indent=2))

    manifest = {
        "job_id": job_id,
        "created_at": datetime.now().isoformat(),
        "subtitle_file": subtitle_path.name,
        "video_file": video_path.name,
        "summary_path": str(summary_path),
        "summary_model": summary_model,
        "clips": [],
    }
    write_clipper_job(job_id, manifest)

    return redirect(url_for('yt.clipper_job', job_id=job_id))


@yt_bp.route('/yt/clipper/job/<job_id>')
def clipper_job(job_id):
    manifest = load_clipper_job(job_id)
    if not manifest:
        flash("Clipper job not found.", "danger")
        return redirect(url_for('yt.clipper'))

    summary_path = Path(manifest.get("summary_path", ""))
    if not summary_path.exists():
        flash("Clipper summary not found.", "danger")
        return redirect(url_for('yt.clipper'))

    try:
        summary_data = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        summary_data = {"segments": [], "error": "Summary JSON is invalid."}

    segments = summary_data.get("segments", [])
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start = segment.get("start")
        end = segment.get("end")
        if not start or not end:
            continue
        try:
            duration_seconds = max(0, (parse_timestamp_to_ms(end) - parse_timestamp_to_ms(start)) / 1000)
        except ValueError:
            continue
        segment["duration_seconds"] = duration_seconds
        segment["duration_label"] = format_duration_seconds(duration_seconds)
    clips_by_segment: dict[int, list[dict]] = {}
    for clip in manifest.get("clips", []):
        seg_index = clip.get("segment_index")
        if isinstance(seg_index, int):
            clips_by_segment.setdefault(seg_index, []).append(clip)
    return render_template(
        'yt/clipper_job.html',
        active_page="clipper",
        manifest=manifest,
        summary=summary_data,
        segments=segments,
        clips_by_segment=clips_by_segment,
    )


@yt_bp.route('/yt/clipper/job/<job_id>/clip', methods=['POST'])
def clipper_clip(job_id):
    manifest = load_clipper_job(job_id)
    if not manifest:
        flash("Clipper job not found.", "danger")
        return redirect(url_for('yt.clipper'))

    summary_path = Path(manifest.get("summary_path", ""))
    if not summary_path.exists():
        flash("Clipper summary not found.", "danger")
        return redirect(url_for('yt.clipper_job', job_id=job_id))

    try:
        summary_data = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        flash("Summary JSON is invalid.", "danger")
        return redirect(url_for('yt.clipper_job', job_id=job_id))

    segments = summary_data.get("segments", [])
    segment_index_raw = (request.form.get("segment_index") or "").strip()
    if not segment_index_raw.isdigit():
        flash("Invalid segment selection.", "danger")
        return redirect(url_for('yt.clipper_job', job_id=job_id))
    segment_index = int(segment_index_raw)
    if segment_index < 0 or segment_index >= len(segments):
        flash("Segment not found.", "danger")
        return redirect(url_for('yt.clipper_job', job_id=job_id))

    segment = segments[segment_index]
    start = segment.get("start")
    end = segment.get("end")
    if not start or not end:
        flash("Segment timestamps are missing.", "danger")
        return redirect(url_for('yt.clipper_job', job_id=job_id))

    video_path = resolve_media_other_file(manifest.get("video_file", ""))
    if not video_path:
        flash("Source video not found in Admin/Media/Other.", "danger")
        return redirect(url_for('yt.clipper_job', job_id=job_id))

    output_filename = (
        f"{video_path.stem}__{format_clip_timestamp(start)}__"
        f"{format_clip_timestamp(end)}__seg_{segment_index + 1:02d}.mp4"
    )
    output_path = MEDIA_CLIPS_DIR / output_filename
    clip_basename = output_path.stem
    mux_mkv_requested = (request.form.get("mux_mkv") or "").strip() == "1"
    subtitle_translate_target = (request.form.get("subtitle_translate_target") or "").strip().lower()
    if subtitle_translate_target not in {"", "en", "it"}:
        subtitle_translate_target = ""

    logger.info(
        "[CLIPPER] clip seg=%s start=%s end=%s -> output=%s",
        segment_index + 1,
        start,
        end,
        output_path,
    )

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                start,
                "-to",
                end,
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        flash("ffmpeg is not installed on the server.", "danger")
        return redirect(url_for('yt.clipper_job', job_id=job_id))

    if result.returncode != 0:
        logger.error("[CLIPPER] ffmpeg error: %s", result.stderr)
        flash(f"Clip generation failed: {result.stderr}", "danger")
        return redirect(url_for('yt.clipper_job', job_id=job_id))

    subtitle_file_raw = str(manifest.get("subtitle_file") or "").strip()
    subtitle_src_path: Path | None = None
    raw_subtitle_path = Path(subtitle_file_raw)
    if raw_subtitle_path.is_absolute() and raw_subtitle_path.exists():
        subtitle_src_path = raw_subtitle_path
    else:
        subtitle_src_path = resolve_media_other_file(subtitle_file_raw)
    if not subtitle_src_path:
        flash("Source subtitle file not found in Admin/Media/Other.", "danger")
        return redirect(url_for('yt.clipper_job', job_id=job_id))

    start_ms = parse_timestamp_to_ms(start)
    end_ms = parse_timestamp_to_ms(end)
    subtitle_filename = f"{clip_basename}.srt"
    subtitle_out_path = MEDIA_CLIPS_DIR / subtitle_filename
    subtitle_vtt_filename: str | None = None
    subtitle_vtt_out_path: Path | None = None
    if subtitle_src_path.suffix.lower() == ".vtt":
        subtitle_vtt_filename = f"{clip_basename}.vtt"
        subtitle_vtt_out_path = MEDIA_CLIPS_DIR / subtitle_vtt_filename

    try:
        cue_count, vtt_written = generate_clip_subtitles(
            sub_src_path=subtitle_src_path,
            clip_start_ms=start_ms,
            clip_end_ms=end_ms,
            out_srt_path=subtitle_out_path,
            out_vtt_path=subtitle_vtt_out_path,
        )
    except ValueError as exc:
        logger.error("[CLIPPER] subtitle generation failed: %s", exc)
        flash("Clip was created, but subtitle generation failed.", "warning")
        cue_count = 0
        vtt_written = False
        subtitle_filename = ""
        subtitle_vtt_filename = None
    else:
        logger.info("[CLIPPER] subtitles generated -> %s cues=%s", subtitle_out_path, cue_count)

    translated_subtitle_filename: str | None = None
    translated_subtitle_vtt_filename: str | None = None
    translated_subtitle_target: str | None = None
    if subtitle_filename and subtitle_translate_target in {"en", "it"}:
        translated_subtitle_filename, translated_subtitle_vtt_filename = maybe_translate_clip_subtitles(
            job_id=job_id,
            segment_index=segment_index,
            subtitle_out_path=subtitle_out_path,
            subtitle_vtt_out_path=subtitle_vtt_out_path if vtt_written else None,
            translate_target=subtitle_translate_target,
        )
        if translated_subtitle_filename:
            translated_subtitle_target = subtitle_translate_target

    mkv_filename: str | None = None
    mkv_success = False
    mkv_out_path = MEDIA_CLIPS_DIR / f"{clip_basename}.mkv"
    if mux_mkv_requested and subtitle_filename:
        mux_ok, mux_error = mux_mkv_ffmpeg(output_path, subtitle_out_path, mkv_out_path)
        mkv_success = mux_ok
        if mux_ok:
            mkv_filename = mkv_out_path.name
        else:
            logger.error("[CLIPPER] mux mkv failed stderr=%s", mux_error)
            flash("Clip and subtitles created, but MKV mux failed.", "warning")
        logger.info(
            "[CLIPPER] mux mkv requested=%s success=%s path=%s",
            mux_mkv_requested,
            mkv_success,
            mkv_out_path,
        )
    else:
        logger.info(
            "[CLIPPER] mux mkv requested=%s success=%s path=%s",
            mux_mkv_requested,
            mkv_success,
            mkv_out_path,
        )

    clip_entry = {
        "segment_index": segment_index,
        "start": start,
        "end": end,
        "clip_filename": output_filename,
        "clip_path": str(output_path),
        "subtitle_filename": subtitle_filename or None,
        "subtitle_vtt_filename": subtitle_vtt_filename if vtt_written else None,
        "translated_subtitle_filename": translated_subtitle_filename,
        "translated_subtitle_vtt_filename": translated_subtitle_vtt_filename,
        "translated_subtitle_target": translated_subtitle_target,
        "mkv_filename": mkv_filename,
        "created_at": datetime.now().isoformat(),
    }
    manifest.setdefault("clips", []).append(clip_entry)
    write_clipper_job(job_id, manifest)
    flash("Clip created successfully.", "success")
    return redirect(url_for('yt.clipper_job', job_id=job_id))


@yt_bp.route('/yt/clipper/download/<path:filename>')
def clipper_download(filename):
    allowed_exts = {".mp4", ".mkv", ".srt", ".vtt"}
    if Path(filename).suffix.lower() not in allowed_exts:
        abort(404)
    resolved = resolve_clips_file(filename)
    if not resolved:
        abort(404)
    return send_from_directory(MEDIA_CLIPS_DIR, resolved.name, as_attachment=True)

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

    return render_template('yt/history.html', history=history, active_page="history")

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

    # Manual test notes:
    # - file with #/## headings splits along those first
    # - file without headings splits by paragraphs
    # - very long subchapter further splits by sentences/hardcut
    # - skip_script ON never triggers OpenAI script generation
    # - switching script_model changes ONLY the script generation request model parameter
    if request.method == 'POST':
        action = request.form.get("action")

        if action == "start":
            files = request.files.getlist("files")
            skip_script = request.form.get("skip_script") in ("on", "true", "1", "yes", "checked")
            auto_split_raw = request.form.get("auto_split")
            auto_split = auto_split_raw in ("on", "true", "1", "yes")
            max_chars_raw = request.form.get("max_chars", "7000")
            try:
                max_chars = clamp_max_chars(int(max_chars_raw))
            except ValueError:
                max_chars = 7000
            logger.info(f"[PODCAST] skip_script={skip_script} raw={request.form.get('skip_script')}")
            logger.debug("[PODCAST] skip_script flag received during start action.")
            if not files or all(not f.filename for f in files):
                flash("Please upload at least one .txt or .md file.", "danger")
            elif len(files) > MAX_PODCAST_FILES:
                flash(f"Please upload no more than {MAX_PODCAST_FILES} files per request.", "danger")
            else:
                job_id = uuid.uuid4().hex[:12]
                job_upload_dir = PODCAST_UPLOAD_DIR / job_id
                job_upload_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "job_id": job_id,
                    "created_at": datetime.now().isoformat(),
                    "uploaded_files": [],
                    "script_files": [],
                    "audio_files": [],
                    "script_generation_state": None,
                    "script_skipped": skip_script,
                    "auto_split": auto_split,
                    "max_chars": max_chars,
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

                    base_name = Path(safe_name).stem
                    suffix = Path(safe_name).suffix.lower()
                    input_name = safe_name
                    input_path = job_upload_dir / input_name
                    if input_path.exists():
                        unique_id = uuid.uuid4().hex[:6]
                        input_name = f"{base_name}_{unique_id}{suffix}"
                        input_path = job_upload_dir / input_name

                    try:
                        upload.save(input_path)
                        manifest["uploaded_files"].append(
                            {
                                "original_name": safe_name,
                                "input_filename": input_name,
                                "input_path": str(input_path),
                                "base_name": base_name,
                                "job_id": job_id,
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
                script_model = request.form.get("script_model", SCRIPT_MODEL)
                if script_model not in {"gpt-4o-mini", "gpt-5-mini"}:
                    script_model = "gpt-4o-mini"
                # Cost note: gpt-5-mini is ~1.67x input and ~3.33x output vs gpt-4o-mini (use intentionally)
                manifest["selected_tone"] = tone
                manifest["selected_script_prompt_name"] = selected_name or manifest.get("selected_script_prompt_name")
                manifest["selected_script_model"] = script_model
                auto_split = bool(manifest.get("auto_split", True))
                max_chars = clamp_max_chars(int(manifest.get("max_chars", 7000)))
                if not skip_script:
                    if not prompt_text and selected_name:
                        prompt = prompt_lookup(script_prompts, selected_name)
                        prompt_text = prompt["text"] if prompt else ""
                    if not prompt_text:
                        flash("Script prompt text is required.", "danger")
                        prompt_text = ""
                if prompt_text or skip_script:
                    try:
                        update_job_status(
                            manifest["job_id"],
                            state="queued",
                            current_file=None,
                            current_chunk=0,
                            total_chunks=0,
                            last_message="Job queued.",
                            errors=[],
                            outputs={},
                        )
                        manifest["script_generation_state"] = "queued"
                        write_job_manifest(manifest["job_id"], manifest)
                        SCRIPT_JOB_EXECUTOR.submit(
                            process_script_job,
                            manifest["job_id"],
                            prompt_text,
                            tone,
                            script_model,
                            skip_script,
                        )
                        flash("Script job started. Progress will update below.", "success")
                    except Exception as e:
                        flash(f"Failed to start script job: {e}", "danger")

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
                            job_audio_dir = PODCAST_AUDIO_DIR / manifest["job_id"]
                            job_audio_dir.mkdir(parents=True, exist_ok=True)
                            audio_parts = []
                            script_path = Path(item.get("script_full_path", ""))
                            if not script_path.exists():
                                errors.append(
                                    {
                                        "file": item["original_name"],
                                        "error": f"Script file missing: {script_path.name}",
                                    }
                                )
                                continue
                            script_text = script_path.read_text(encoding="utf-8-sig", errors="ignore")
                            if not script_text.strip():
                                errors.append(
                                    {
                                        "file": item["original_name"],
                                        "error": f"Script file empty: {script_path.name}",
                                    }
                                )
                                continue
                            auto_split = bool(item.get("auto_split", True))
                            max_chars = clamp_max_chars(int(item.get("max_chars", 7000)))
                            tts_chunks = split_for_tts(script_text, max_chars=max_chars) if auto_split else [script_text]
                            logger.info(
                                "[TTS] splitting full script into %s chunks, max_chars=%s",
                                len(tts_chunks),
                                max_chars,
                            )
                            update_job_status(
                                manifest["job_id"],
                                state="running",
                                current_file=item["original_name"],
                                current_chunk=0,
                                total_chunks=len(tts_chunks),
                                last_message=f"TTS chunk 0/{len(tts_chunks)} for {item['original_name']}.",
                            )
                            log_chunk_warning(manifest["job_id"], item["original_name"], len(tts_chunks), max_chars)
                            for index, chunk in enumerate(tts_chunks, start=1):
                                update_job_status(
                                    manifest["job_id"],
                                    current_chunk=index,
                                    last_message=f"TTS chunk {index}/{len(tts_chunks)} for {item['original_name']}.",
                                )
                                try:
                                    audio_name = f"{item['base_name']}_part_{index:02d}.mp3"
                                    audio_path = job_audio_dir / audio_name
                                    audio_bytes = tts_generate_audio(client, chunk, voice, prompt_text)
                                    audio_path.write_bytes(audio_bytes)
                                    audio_parts.append(
                                        {
                                            "audio_filename": f"{manifest['job_id']}/{audio_name}",
                                            "audio_path": str(audio_path),
                                        }
                                    )
                                    logger.info("[TTS] part %02d/%02d OK -> %s", index, len(tts_chunks), audio_path)
                                except Exception as e:
                                    errors.append(
                                        {
                                            "file": item["original_name"],
                                            "error": f"TTS part {index:02d}: {e}",
                                        }
                                    )
                            audio_full_name = f"{item['base_name']}_full.mp3"
                            audio_full_path = job_audio_dir / audio_full_name
                            merge_ok, merge_error = merge_mp3_parts_ffmpeg(
                                [Path(part["audio_path"]) for part in audio_parts],
                                audio_full_path,
                            )
                            if not merge_ok:
                                if merge_error:
                                    errors.append({"file": item["original_name"], "error": merge_error})
                                audio_full_name = None
                            else:
                                logger.info("[TTS] merge OK -> %s", audio_full_path)
                            manifest["audio_files"].append(
                                {
                                    "original_name": item["original_name"],
                                    "audio_parts": audio_parts,
                                    "audio_full_filename": f"{manifest['job_id']}/{audio_full_name}" if audio_full_name else None,
                                    "audio_full_path": str(audio_full_path) if audio_full_name else None,
                                    "script_full_filename": item.get("script_full_filename"),
                                    "script_skipped": item.get("script_skipped", False),
                                    "chunk_count": len(audio_parts),
                                    "auto_split": item.get("auto_split", False),
                                    "max_chars": item.get("max_chars"),
                                    "headings_detected": item.get("headings_detected", False),
                                    "headings_used": item.get("headings_used", False),
                                    "chunk_strategy": item.get("chunk_strategy"),
                                    "script_model": item.get("script_model"),
                                }
                            )
                        write_job_manifest(manifest["job_id"], manifest)

    if manifest:
        skip_script_checked = bool(manifest.get("script_skipped"))
    elif request.method == "POST":
        skip_script_checked = request.form.get("skip_script") in ("on", "true", "1", "yes", "checked")
    job_status = load_job_status(job_id) if job_id else None

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
        job_status=job_status,
        smb_base_url=SMB_BASE_URL.rstrip("/"),
        active_page="podcast",
    )


@yt_bp.route('/open-folder/<job_id>/<kind>')
def open_folder(job_id, kind):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", job_id):
        flash("Invalid job id.", "danger")
        return redirect(url_for('yt.podcast'))
    job_dir = resolve_job_folder(job_id, kind)
    if not job_dir:
        flash("Invalid folder kind.", "danger")
        return redirect(url_for('yt.podcast', job_id=job_id))
    if not job_dir.exists() or not job_dir.is_dir():
        flash(f"Folder not found for job_id={job_id}.", "danger")
        return redirect(url_for('yt.podcast', job_id=job_id))
    logger.info("[UI] open-folder job_id=%s kind=%s path=%s", job_id, kind, job_dir)
    if sys.platform == "darwin":
        subprocess.run(["open", str(job_dir)], check=False)
        flash(f"Opened {kind} folder for job_id={job_id}", "success")
    elif shutil.which("xdg-open"):
        subprocess.run(["xdg-open", str(job_dir)], check=False)
        flash(f"Opened {kind} folder for job_id={job_id}", "success")
    else:
        flash("Folder opening not supported on this OS.", "danger")
    return redirect(url_for('yt.podcast', job_id=job_id))


@yt_bp.route('/job/<job_id>/status')
def podcast_job_status(job_id):
    status = load_job_status(job_id)
    if not status:
        return {"state": "unknown", "last_message": "Status not found."}, 404
    return status


@yt_bp.route('/yt/podcast/download/<category>/<path:filename>')
def podcast_download(category, filename):
    directories = {
        "scripts": PODCAST_SCRIPT_DIR,
        "audio": PODCAST_AUDIO_DIR,
        "uploads": PODCAST_UPLOAD_DIR,
    }
    if category not in directories:
        abort(404)
    file_path = Path(filename)
    if file_path.is_absolute() or ".." in file_path.parts:
        abort(404)
    return send_from_directory(directories[category], filename, as_attachment=True)


@yt_bp.route('/yt/podcast/media/<path:filename>')
def podcast_media(filename):
    file_path = Path(filename)
    if file_path.is_absolute() or ".." in file_path.parts:
        abort(404)
    return send_from_directory(PODCAST_AUDIO_DIR, filename, as_attachment=False)

# ------- Magnet link downloader -------
@yt_bp.route('/yt/magnet', methods=['GET', 'POST'])
def magnet_download():
    if request.method == 'GET':
        return render_template('yt/magnet.html', folders=DOWNLOAD_FOLDERS, active_page="magnet")

    magnet = request.form.get('magnet', '').strip()
    folder_key = request.form.get('folder', '')
    folder_path = DOWNLOAD_FOLDERS.get(folder_key, '/tmp')

    if not magnet:
        flash('Magnet link required', 'danger')
        return redirect(url_for('yt.magnet_download'))

    uid = str(uuid.uuid4())[:8]

    # Verify that aria2c is available before attempting the download
    if shutil.which('aria2c') is None:
        flash('aria2c is required for magnet downloads. Please install it first.', 'danger')
        return redirect(url_for('yt.magnet_download'))

    cmd = ['aria2c', '--dir', folder_path, magnet]

    try:
        Thread(target=run_download_and_track, args=(cmd, uid), daemon=True).start()
        log_magnet_download(magnet, folder_path)
        flash(f"Magnet download started [ID: {uid}]", 'success')
    except Exception as e:
        flash(f"Magnet download error: {e}", 'danger')

    return redirect(url_for('yt.magnet_download'))
