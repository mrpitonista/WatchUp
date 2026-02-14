from __future__ import annotations

import re
import json
import html
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


@dataclass
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str


TIME_SPLIT_RE = re.compile(r"\s+-->\s+")
SRT_TIMESTAMP_RE = re.compile(r"(\d{1,2}:\d{2}:\d{2}[\.,]\d{1,3})")
VTT_INLINE_TIMESTAMP_RE = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")
MIN_CUE_MS = 200
MERGE_GAP_MS = 120


def parse_timestamp_to_ms(value: str) -> int:
    normalized = value.strip().replace(",", ".")
    parts = normalized.split(":")
    if len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds_part = parts[1]
    elif len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_part = parts[2]
    else:
        raise ValueError(f"Unsupported timestamp format: {value}")
    if "." in seconds_part:
        seconds_str, millis_str = seconds_part.split(".", 1)
        millis = int(millis_str.ljust(3, "0")[:3])
    else:
        seconds_str = seconds_part
        millis = 0
    seconds = int(seconds_str)
    total_ms = ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis
    return total_ms


def ms_to_timestamp(ms: int) -> str:
    total_seconds, millis = divmod(int(ms), 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def format_srt_time(ms: int) -> str:
    total_seconds, millis = divmod(max(0, int(ms)), 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_vtt_time(ms: int) -> str:
    total_seconds, millis = divmod(max(0, int(ms)), 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def write_srt(cues: list[SubtitleCue], path: Path) -> None:
    lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.extend(
            [
                str(index),
                f"{format_srt_time(cue.start_ms)} --> {format_srt_time(cue.end_ms)}",
                cue.text,
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_vtt(cues: list[SubtitleCue], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.extend(
            [
                f"{format_vtt_time(cue.start_ms)} --> {format_vtt_time(cue.end_ms)}",
                cue.text,
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def normalize_text(lines: Iterable[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    return re.sub(r"\s+", " ", text).strip()


def clean_vtt_text(text: str) -> str:
    cleaned = VTT_INLINE_TIMESTAMP_RE.sub("", text)
    cleaned = re.sub(r"<c(\.[^>]*)?>", "", cleaned)
    cleaned = re.sub(r"</c>", "", cleaned)
    cleaned = re.sub(r"<v[^>]*>", "", cleaned)
    cleaned = re.sub(r"</v>", "", cleaned)
    cleaned = re.sub(r"</?(?:i|b|u)>", "", cleaned)
    cleaned = re.sub(r"</?[^>]+>", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def find_verdana_ttf() -> Path | None:
    candidates = [
        Path("/Library/Fonts/Verdana.ttf"),
        Path("/System/Library/Fonts/Supplemental/Verdana.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def build_transcript_paragraphs(cues: list[SubtitleCue]) -> list[dict[str, str]]:
    paragraphs: list[dict[str, str]] = []
    current_lines: list[str] = []
    current_start_ms: int | None = None
    current_length = 0
    previous_end_ms: int | None = None

    for cue in cues:
        text = clean_vtt_text(html.unescape(cue.text.strip()))
        if not text:
            continue

        gap_ms = cue.start_ms - previous_end_ms if previous_end_ms is not None else 0
        should_split = (
            bool(current_lines)
            and (
                gap_ms > 1200
                or current_length + len(text) > 850
            )
        )
        if should_split and current_start_ms is not None:
            paragraphs.append(
                {
                    "ts": ms_to_timestamp(current_start_ms).split(".")[0],
                    "text": " ".join(current_lines).strip(),
                }
            )
            current_lines = []
            current_length = 0
            current_start_ms = None

        if current_start_ms is None:
            current_start_ms = cue.start_ms
        current_lines.append(text)
        current_length += len(text)
        previous_end_ms = cue.end_ms

    if current_lines and current_start_ms is not None:
        paragraphs.append(
            {
                "ts": ms_to_timestamp(current_start_ms).split(".")[0],
                "text": " ".join(current_lines).strip(),
            }
        )

    return [p for p in paragraphs if p["text"]]


def generate_transcript_pdf(
    *,
    title: str,
    video_filename: str,
    subtitle_filename: str,
    duration_label: str,
    source_language: str,
    job_id: str,
    cues: list[SubtitleCue],
    out_pdf_path: Path,
    generated_at: datetime | None = None,
    verdana_path: Path | None = None,
) -> int:
    if not cues:
        raise ValueError("No subtitle cues found for transcript generation.")

    if verdana_path is None:
        verdana_path = find_verdana_ttf()
    if verdana_path is None:
        raise FileNotFoundError("Verdana font not found.")

    pdfmetrics.registerFont(TTFont("Verdana", str(verdana_path)))
    pdfmetrics.registerFont(TTFont("Verdana-Bold", str(verdana_path)))

    paragraphs = build_transcript_paragraphs(cues)
    if not paragraphs:
        raise ValueError("No transcript paragraphs generated from subtitle cues.")

    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_pdf_path),
        pagesize=LETTER,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=title,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TranscriptTitle",
        parent=styles["Title"],
        fontName="Verdana-Bold",
        fontSize=18,
        leading=24,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "TranscriptMeta",
        parent=styles["Normal"],
        fontName="Verdana",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#374151"),
        spaceAfter=2,
    )
    label_style = ParagraphStyle(
        "TranscriptLabel",
        parent=styles["Normal"],
        fontName="Verdana",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "TranscriptBody",
        parent=styles["Normal"],
        fontName="Verdana",
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#111827"),
        spaceAfter=10,
    )

    created_at = generated_at or datetime.now()
    header_rows = [
        f"<b>Video filename:</b> {html.escape(video_filename)}",
        f"<b>Subtitle filename:</b> {html.escape(subtitle_filename)}",
        f"<b>Duration:</b> {html.escape(duration_label)}",
        f"<b>Source language:</b> {html.escape(source_language)}",
        f"<b>Job ID:</b> {html.escape(job_id)}",
        f"<b>Generated:</b> {created_at.strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    story: list = [
        Paragraph(html.escape(title), title_style),
        Spacer(1, 4),
    ]
    for row in header_rows:
        story.append(Paragraph(row, meta_style))
    story.append(Spacer(1, 12))

    for paragraph in paragraphs:
        label = f"[{paragraph['ts']}]"
        body = html.escape(paragraph["text"])
        story.append(Paragraph(label, label_style))
        story.append(Paragraph(body, body_style))

    def _draw_page_number(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Verdana", 9)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(LETTER[0] - 0.8 * inch, 0.5 * inch, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_page_number, onLaterPages=_draw_page_number)
    return len(paragraphs)


def parse_srt(path: Path) -> list[SubtitleCue]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", content.strip())
    cues: list[SubtitleCue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        if SRT_TIMESTAMP_RE.search(lines[0]):
            time_line = lines[0]
            text_lines = lines[1:]
        else:
            if len(lines) < 3:
                continue
            time_line = lines[1]
            text_lines = lines[2:]
        match = TIME_SPLIT_RE.split(time_line)
        if len(match) != 2:
            continue
        try:
            start_ms = parse_timestamp_to_ms(match[0])
            end_ms = parse_timestamp_to_ms(match[1].split()[0])
        except ValueError:
            continue
        text = normalize_text(text_lines)
        if not text:
            continue
        cues.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text))
    return cues


def parse_vtt(path: Path) -> list[SubtitleCue]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.rstrip("\n") for line in content.splitlines()]
    cues: list[SubtitleCue] = []
    buffer_lines: list[str] = []
    time_line: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if time_line:
                match = TIME_SPLIT_RE.split(time_line)
                if len(match) == 2:
                    try:
                        start_ms = parse_timestamp_to_ms(match[0])
                        end_ms = parse_timestamp_to_ms(match[1].split()[0])
                        text = normalize_text(buffer_lines)
                        text = clean_vtt_text(text)
                        if text:
                            cues.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text))
                    except ValueError:
                        pass
            buffer_lines = []
            time_line = None
            continue
        if stripped.startswith("WEBVTT"):
            continue
        if "-->" in stripped:
            time_line = stripped
            buffer_lines = []
            continue
        if time_line:
            buffer_lines.append(stripped)
    if time_line and buffer_lines:
        match = TIME_SPLIT_RE.split(time_line)
        if len(match) == 2:
            try:
                start_ms = parse_timestamp_to_ms(match[0])
                end_ms = parse_timestamp_to_ms(match[1].split()[0])
                text = normalize_text(buffer_lines)
                text = clean_vtt_text(text)
                if text:
                    cues.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text))
            except ValueError:
                pass
    return cues


def merge_adjacent_duplicate_cues(cues: list[SubtitleCue], merge_gap_ms: int = MERGE_GAP_MS) -> list[SubtitleCue]:
    if not cues:
        return []
    merged: list[SubtitleCue] = [cues[0]]
    for cue in cues[1:]:
        previous = merged[-1]
        gap_ms = cue.start_ms - previous.end_ms
        prev_text = previous.text.strip()
        cue_text = cue.text.strip()
        same_or_prefix = (
            prev_text == cue_text
            or prev_text.startswith(cue_text)
            or cue_text.startswith(prev_text)
        )
        if gap_ms <= merge_gap_ms and same_or_prefix:
            merged[-1] = SubtitleCue(
                start_ms=previous.start_ms,
                end_ms=max(previous.end_ms, cue.end_ms),
                text=prev_text if len(prev_text) >= len(cue_text) else cue_text,
            )
            continue
        merged.append(cue)
    return merged


def generate_clip_subtitles(
    sub_src_path: Path,
    clip_start_ms: int,
    clip_end_ms: int,
    out_srt_path: Path,
    out_vtt_path: Path | None = None,
) -> tuple[int, bool]:
    suffix = sub_src_path.suffix.lower()
    if suffix == ".srt":
        source_cues = parse_srt(sub_src_path)
    elif suffix == ".vtt":
        source_cues = parse_vtt(sub_src_path)
    else:
        raise ValueError("Unsupported subtitle source format.")

    clipped_cues: list[SubtitleCue] = []
    for cue in source_cues:
        if cue.end_ms <= clip_start_ms or cue.start_ms >= clip_end_ms:
            continue
        cue_start = max(cue.start_ms, clip_start_ms)
        cue_end = min(cue.end_ms, clip_end_ms)
        if cue_end <= cue_start:
            continue
        if (cue_end - cue_start) < MIN_CUE_MS:
            continue
        clipped_cues.append(
            SubtitleCue(
                start_ms=cue_start - clip_start_ms,
                end_ms=cue_end - clip_start_ms,
                text=cue.text,
            )
        )

    clipped_cues = merge_adjacent_duplicate_cues(clipped_cues)

    write_srt(clipped_cues, out_srt_path)
    vtt_written = False
    if out_vtt_path is not None:
        write_vtt(clipped_cues, out_vtt_path)
        vtt_written = True
    return len(clipped_cues), vtt_written


def get_google_translate_client(credentials_path: Path) -> translate.Client:
    credentials = service_account.Credentials.from_service_account_file(str(credentials_path))
    return translate.Client(credentials=credentials)


def detect_language_text(client: translate.Client, text: str) -> str:
    if not text.strip():
        return ""
    detected = client.detect_language(text)
    if isinstance(detected, list) and detected:
        detected = detected[0]
    if isinstance(detected, dict):
        return str(detected.get("language") or "").lower()
    return ""


def translate_texts(
    client: translate.Client,
    texts: list[str],
    target: str,
    batch_size: int = 100,
) -> list[str]:
    translated: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.translate(batch, target_language=target, format_="text")
        if isinstance(response, dict):
            response = [response]
        translated.extend(html.unescape(str(item.get("translatedText") or "")) for item in response)
    return translated


def mux_mkv_ffmpeg(clip_mp4_path: Path, srt_path: Path, out_mkv_path: Path) -> tuple[bool, str]:
    first_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(clip_mp4_path),
        "-i",
        str(srt_path),
        "-c",
        "copy",
        "-c:s",
        "srt",
        str(out_mkv_path),
    ]
    fallback_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(clip_mp4_path),
        "-i",
        str(srt_path),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-c:s",
        "srt",
        str(out_mkv_path),
    ]
    first_run = subprocess.run(first_cmd, capture_output=True, text=True, check=False)
    if first_run.returncode == 0:
        return True, first_run.stderr.strip()

    fallback_run = subprocess.run(fallback_cmd, capture_output=True, text=True, check=False)
    if fallback_run.returncode == 0:
        return True, (fallback_run.stderr.strip() or first_run.stderr.strip())
    error_text = fallback_run.stderr.strip() or first_run.stderr.strip() or "ffmpeg mux failed."
    return False, error_text


def build_blocks(
    cues: list[SubtitleCue],
    target_seconds: int = 45,
    max_chars: int = 1400,
) -> list[dict]:
    if not cues:
        return []
    blocks: list[dict] = []
    current_text: list[str] = []
    block_start = cues[0].start_ms
    block_end = cues[0].end_ms
    for cue in cues:
        if not current_text:
            block_start = cue.start_ms
            block_end = cue.end_ms
        else:
            block_end = max(block_end, cue.end_ms)
        current_text.append(cue.text)
        joined = " ".join(current_text)
        duration = (block_end - block_start) / 1000
        if duration >= target_seconds or len(joined) >= max_chars:
            blocks.append(
                {
                    "start_ms": block_start,
                    "end_ms": block_end,
                    "text": joined.strip(),
                }
            )
            current_text = []
    if current_text:
        blocks.append(
            {
                "start_ms": block_start,
                "end_ms": block_end,
                "text": " ".join(current_text).strip(),
            }
        )
    return blocks


def blocks_to_prompt_text(blocks: list[dict]) -> str:
    lines: list[str] = []
    for block in blocks:
        start = ms_to_timestamp(block["start_ms"])
        end = ms_to_timestamp(block["end_ms"])
        text = block["text"].strip()
        lines.append(f"[{start}–{end}] {text}")
    return "\n".join(lines)


def safe_list_media_files(directory: Path, exts: Iterable[str]) -> list[str]:
    allowed = {ext.lower() for ext in exts}
    files: list[str] = []
    if not directory.exists():
        return files
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in allowed:
            files.append(item.name)
    return sorted(files)


def find_matching_video(sub_path: Path) -> str | None:
    video_exts = {".mp4", ".mkv", ".webm", ".mov"}
    directory = sub_path.parent
    stem = sub_path.stem
    for ext in video_exts:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate.name
    return None


def list_clipper_jobs(jobs_dir: Path, summary_dir: Path, limit: int = 15) -> list[dict]:
    jobs: list[dict] = []
    if not jobs_dir.exists():
        return jobs

    for manifest_path in jobs_dir.glob("*.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue

        job_id = str(manifest.get("job_id") or manifest_path.stem)
        created_at = manifest.get("created_at")
        created_at_dt: datetime | None = None
        if isinstance(created_at, str):
            try:
                created_at_dt = datetime.fromisoformat(created_at)
            except ValueError:
                created_at_dt = None

        summary_data: dict = {}
        summary_path_raw = manifest.get("summary_path")
        if isinstance(summary_path_raw, str) and summary_path_raw:
            summary_path = Path(summary_path_raw).resolve()
            try:
                summary_path.relative_to(summary_dir.resolve())
            except ValueError:
                summary_path = None
            if summary_path and summary_path.exists():
                try:
                    loaded_summary = json.loads(summary_path.read_text())
                    if isinstance(loaded_summary, dict):
                        summary_data = loaded_summary
                except (OSError, json.JSONDecodeError):
                    summary_data = {}

        subtitle_file = manifest.get("subtitle_file") or "Unknown"
        video_file = manifest.get("video_file") or "Unknown"
        language = manifest.get("language") or summary_data.get("language") or "Unknown"
        model_used = manifest.get("model") or manifest.get("summary_model") or "Unknown"
        title = (
            manifest.get("title")
            or summary_data.get("title")
            or Path(str(video_file)).stem
            or "Unknown"
        )

        sort_value = created_at_dt.timestamp() if created_at_dt else manifest_path.stat().st_mtime

        jobs.append(
            {
                "job_id": job_id,
                "created_at": created_at if isinstance(created_at, str) else None,
                "subtitle_file": str(subtitle_file),
                "video_file": str(video_file),
                "language": str(language),
                "model": str(model_used),
                "title": str(title),
                "_sort_value": sort_value,
            }
        )

    jobs.sort(key=lambda item: item["_sort_value"], reverse=True)
    for job in jobs:
        job.pop("_sort_value", None)
    return jobs[:limit]
