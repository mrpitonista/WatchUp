from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str


TIME_SPLIT_RE = re.compile(r"\s+-->\s+")
SRT_TIMESTAMP_RE = re.compile(r"(\d{1,2}:\d{2}:\d{2}[\.,]\d{1,3})")


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


def normalize_text(lines: Iterable[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    return re.sub(r"\s+", " ", text).strip()


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
                if text:
                    cues.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text))
            except ValueError:
                pass
    return cues


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
