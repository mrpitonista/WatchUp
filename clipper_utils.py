from __future__ import annotations

import re
import json
import html
import subprocess
import logging
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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


logger = logging.getLogger(__name__)


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


def _normalize_merge_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line).strip()


def merge_text_two_lines(
    a: str,
    b: str,
    max_lines: int = 2,
    max_chars_per_line: int = 120,
) -> str:
    left = _normalize_merge_text(a)
    right = _normalize_merge_text(b)
    if not left:
        return right
    if not right:
        return left

    left_lower = left.lower()
    right_lower = right.lower()
    if left_lower in right_lower:
        merged = right if len(right) >= len(left) else left
    elif right_lower in left_lower:
        merged = left if len(left) >= len(right) else right
    else:
        lines: list[str] = []
        for line in left.split("\n") + right.split("\n"):
            clean_line = re.sub(r"\s+", " ", line).strip()
            if not clean_line:
                continue
            if clean_line not in lines:
                lines.append(clean_line)

        merged_lines = lines[: max(1, max_lines)]
        merged = "\n".join(merged_lines)

    if max_chars_per_line > 0:
        constrained_lines = []
        for line in merged.split("\n"):
            constrained_lines.append(line[:max_chars_per_line].rstrip())
        merged = "\n".join(line for line in constrained_lines if line)
    return merged.strip()


def merge_overlapping_cues(
    cues: list[SubtitleCue | dict | tuple],
    max_lines: int = 2,
    max_chars_per_line: int = 120,
    gap_merge_ms: int = 250,
    max_group_duration_ms: int = 12000,
    max_overlap_merge_ms: int = 3000,
    max_group_cues: int = 5,
) -> list[SubtitleCue]:
    normalized: list[SubtitleCue] = []
    for cue in cues:
        if isinstance(cue, SubtitleCue):
            normalized.append(cue)
            continue
        if isinstance(cue, dict):
            start_ms = int(cue.get("start_ms", 0))
            end_ms = int(cue.get("end_ms", 0))
            text = str(cue.get("text", ""))
            normalized.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text))
            continue
        if isinstance(cue, tuple) and len(cue) >= 3:
            start_ms, end_ms, text = cue[:3]
            normalized.append(SubtitleCue(start_ms=int(start_ms), end_ms=int(end_ms), text=str(text)))

    if not normalized:
        return []

    ordered = sorted(normalized, key=lambda item: (item.start_ms, item.end_ms))
    merged: list[SubtitleCue] = []
    i = 0
    while i < len(ordered):
        current = ordered[i]
        group_cues = 1
        while i + 1 < len(ordered):
            nxt = ordered[i + 1]
            if nxt.start_ms >= current.end_ms:
                break

            overlap_ms = current.end_ms - nxt.start_ms
            new_start = min(current.start_ms, nxt.start_ms)
            new_end = max(current.end_ms, nxt.end_ms)
            merged_duration_ms = new_end - new_start
            if (
                overlap_ms > max(0, int(max_overlap_merge_ms))
                or merged_duration_ms > max(1, int(max_group_duration_ms))
                or (group_cues + 1) > max(1, int(max_group_cues))
            ):
                break

            current = SubtitleCue(
                start_ms=new_start,
                end_ms=new_end,
                text=merge_text_two_lines(
                    current.text,
                    nxt.text,
                    max_lines=max_lines,
                    max_chars_per_line=max_chars_per_line,
                ),
            )
            group_cues += 1
            i += 1
        if current.end_ms > current.start_ms:
            merged.append(current)
        i += 1

    if len(merged) < 2:
        return merged

    normalized_gap = max(0, int(gap_merge_ms))
    non_overlapping: list[SubtitleCue] = [merged[0]]
    for cue in merged[1:]:
        previous = non_overlapping[-1]
        if (
            previous.end_ms > cue.start_ms
            and (previous.end_ms - cue.start_ms) <= normalized_gap
            and cue.start_ms > previous.start_ms
        ):
            clamped_end = max(previous.start_ms + 1, min(previous.end_ms, cue.start_ms))
            non_overlapping[-1] = SubtitleCue(
                start_ms=previous.start_ms,
                end_ms=clamped_end,
                text=previous.text,
            )
        if cue.end_ms > cue.start_ms:
            non_overlapping.append(cue)
    return non_overlapping


def resolve_overlaps_by_trimming(
    cues: list[SubtitleCue | dict | tuple],
    epsilon_ms: int = 1,
    min_duration_ms: int = 250,
) -> list[SubtitleCue]:
    normalized: list[SubtitleCue] = []
    for cue in cues:
        if isinstance(cue, SubtitleCue):
            normalized.append(SubtitleCue(start_ms=int(cue.start_ms), end_ms=int(cue.end_ms), text=str(cue.text)))
            continue
        if isinstance(cue, dict):
            normalized.append(
                SubtitleCue(
                    start_ms=int(cue.get("start_ms", 0)),
                    end_ms=int(cue.get("end_ms", 0)),
                    text=str(cue.get("text", "")),
                )
            )
            continue
        if isinstance(cue, tuple) and len(cue) >= 3:
            start_ms, end_ms, text = cue[:3]
            normalized.append(SubtitleCue(start_ms=int(start_ms), end_ms=int(end_ms), text=str(text)))

    if not normalized:
        logger.info("[CLIPPER] trim-overlap before=0 after=0 adjusted=0 dropped=0")
        return []

    epsilon_ms = max(0, int(epsilon_ms))
    min_duration_ms = max(1, int(min_duration_ms))
    ordered = sorted(normalized, key=lambda item: (item.start_ms, item.end_ms))
    original_ends = [cue.end_ms for cue in ordered]

    overlaps_before = sum(1 for i in range(len(ordered) - 1) if ordered[i].end_ms > ordered[i + 1].start_ms)
    adjusted_indexes: set[int] = set()

    for index in range(len(ordered) - 1):
        cur = ordered[index]
        nxt = ordered[index + 1]
        if cur.end_ms <= nxt.start_ms:
            continue

        latest_allowed_end = nxt.start_ms - epsilon_ms
        new_end = max(cur.start_ms + min_duration_ms, latest_allowed_end)
        if new_end > latest_allowed_end:
            new_end = max(cur.start_ms + 1, latest_allowed_end)

        new_end = min(new_end, original_ends[index])
        if new_end > latest_allowed_end:
            new_end = latest_allowed_end

        if new_end <= cur.start_ms:
            new_end = min(max(cur.start_ms + 1, 0), original_ends[index])

        if new_end != cur.end_ms:
            cur.end_ms = int(new_end)
            adjusted_indexes.add(index)

    kept: list[SubtitleCue] = []
    dropped_count = 0
    for cue in ordered:
        if cue.end_ms <= cue.start_ms:
            dropped_count += 1
            continue
        kept.append(cue)

    for index in range(len(kept) - 1):
        cur = kept[index]
        nxt = kept[index + 1]
        if cur.end_ms <= nxt.start_ms:
            continue
        clamped_end = max(cur.start_ms + 1, nxt.start_ms - epsilon_ms)
        if clamped_end != cur.end_ms:
            cur.end_ms = clamped_end
            adjusted_indexes.add(index)

    final_cues: list[SubtitleCue] = []
    for cue in kept:
        if cue.end_ms <= cue.start_ms:
            dropped_count += 1
            continue
        final_cues.append(cue)

    overlaps_after = sum(1 for i in range(len(final_cues) - 1) if final_cues[i].end_ms > final_cues[i + 1].start_ms)
    logger.info(
        "[CLIPPER] trim-overlap before=%s after=%s adjusted=%s dropped=%s epsilon_ms=%s min_duration_ms=%s",
        overlaps_before,
        overlaps_after,
        len(adjusted_indexes),
        dropped_count,
        epsilon_ms,
        min_duration_ms,
    )
    return final_cues


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


def generate_summary_pdf(job_meta: dict, summary: dict, out_path: Path) -> None:
    verdana_path = find_verdana_ttf()
    base_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    if verdana_path:
        pdfmetrics.registerFont(TTFont("Verdana", str(verdana_path)))
        pdfmetrics.registerFont(TTFont("Verdana-Bold", str(verdana_path)))
        base_font = "Verdana"
        bold_font = "Verdana-Bold"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=str(job_meta.get("title") or "Summary"),
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SummaryPdfTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "SummaryPdfMeta",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#374151"),
    )
    table_header_style = ParagraphStyle(
        "SummaryPdfTableHeader",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#111827"),
    )
    segment_style = ParagraphStyle(
        "SummaryPdfSegment",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#111827"),
    )
    summary_style = ParagraphStyle(
        "SummaryPdfSummary",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#111827"),
    )
    meta_rows = [
        ("Job ID", str(job_meta.get("job_id") or "Unknown")),
        ("Subtitle", str(job_meta.get("subtitle") or "Unknown")),
        ("Video", str(job_meta.get("video") or "Unknown")),
        ("Duration", str(job_meta.get("duration") or "Unknown")),
        ("Language", str(job_meta.get("language") or "Unknown")),
        ("Original language", str(job_meta.get("original_language") or "Unknown")),
        ("Sections identified", str(job_meta.get("sections_identified") or 0)),
        ("Model used", str(job_meta.get("model_used") or "Unknown")),
        ("Source URL", str(job_meta.get("source_url") or "Unknown")),
        ("Source UID", str(job_meta.get("source_uid") or "Unknown")),
    ]

    title = html.escape(str(job_meta.get("title") or "Summary").strip() or "Summary")
    story: list = [Paragraph(title, title_style), Spacer(1, 4)]
    for label, value in meta_rows:
        story.append(Paragraph(f"<b>{html.escape(label)}:</b> {html.escape(value)}", meta_style))
    story.append(Spacer(1, 12))

    segments = summary.get("segments", []) if isinstance(summary, dict) else []
    if not isinstance(segments, list):
        segments = []

    table_data: list[list[Paragraph]] = [
        [Paragraph("Segment", table_header_style), Paragraph("Summary", table_header_style)]
    ]
    for idx, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue
        start = html.escape(str(segment.get("start") or "Unknown"))
        end = html.escape(str(segment.get("end") or "Unknown"))
        duration = html.escape(str(segment.get("duration_label") or ""))
        headline = html.escape(str(segment.get("headline") or f"Segment {idx}"))
        summary_text = html.escape(str(segment.get("summary") or ""))
        why = html.escape(str(segment.get("why_it_matters") or ""))

        left_parts = [f"{start} &#8594; {end}"]
        if duration:
            left_parts.append(duration)
        left_parts.append(f"<b>{headline}</b>")
        left_cell = Paragraph("<br/>".join(left_parts), segment_style)

        right_parts = [f"<para>{summary_text}</para>"]
        if why:
            right_parts.append(
                f"<para><font color='#6b7280' size='9'>Why it matters: {why}</font></para>"
            )
        right_cell = Paragraph("".join(right_parts), summary_style)
        table_data.append([left_cell, right_cell])

    usable_width = LETTER[0] - doc.leftMargin - doc.rightMargin
    segments_table = Table(
        table_data,
        colWidths=[usable_width * 0.3, usable_width * 0.7],
        repeatRows=1,
    )
    segments_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#d1d5db")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(segments_table)

    def _draw_page_number(canvas, _doc):
        canvas.saveState()
        canvas.setFont(base_font, 9)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(LETTER[0] - doc.rightMargin, 0.45 * inch, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_page_number, onLaterPages=_draw_page_number)


def generate_transcriber_summary_pdf(job_meta: dict, summary_text: str, out_path: Path) -> None:
    verdana_path = find_verdana_ttf()
    base_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    if verdana_path:
        pdfmetrics.registerFont(TTFont("Verdana", str(verdana_path)))
        pdfmetrics.registerFont(TTFont("Verdana-Bold", str(verdana_path)))
        base_font = "Verdana"
        bold_font = "Verdana-Bold"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=str(job_meta.get("title") or "Summary"),
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TranscriberSummaryTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "TranscriberSummaryMeta",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#374151"),
    )
    section_style = ParagraphStyle(
        "TranscriberSummarySection",
        parent=styles["Normal"],
        fontName=bold_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#111827"),
    )
    summary_style = ParagraphStyle(
        "TranscriberSummaryBody",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#111827"),
    )

    meta_rows = [
        ("Job ID", str(job_meta.get("job_id") or "Unknown")),
        ("Audio file", str(job_meta.get("audio_file") or "Unknown")),
        ("Duration", str(job_meta.get("duration") or "Unknown")),
        ("Detected language", str(job_meta.get("language") or "Unknown")),
        ("Transcription model", str(job_meta.get("transcription_model") or "Unknown")),
        ("Summary source", str(job_meta.get("summary_source") or "Unknown")),
        ("Summary model", str(job_meta.get("summary_model") or "Unknown")),
        ("Created", str(job_meta.get("created_at") or "Unknown")),
    ]

    title = html.escape(str(job_meta.get("title") or "Summary").strip() or "Summary")
    story: list = [Paragraph(title, title_style), Spacer(1, 4)]
    for label, value in meta_rows:
        story.append(Paragraph(f"<b>{html.escape(label)}:</b> {html.escape(value)}", meta_style))
    story.append(Spacer(1, 12))

    paragraphs = [p.strip() for p in summary_text.split("\n\n") if p.strip()]
    table_data: list[list[Paragraph]] = [
        [Paragraph("Section", section_style), Paragraph("Summary", section_style)]
    ]
    for idx, paragraph in enumerate(paragraphs, start=1):
        cleaned = html.escape(paragraph).replace("\n", "<br/>")
        table_data.append(
            [
                Paragraph(f"Section {idx}", summary_style),
                Paragraph(cleaned, summary_style),
            ]
        )

    if len(table_data) == 1:
        table_data.append([Paragraph("Section 1", summary_style), Paragraph("(empty)", summary_style)])

    usable_width = LETTER[0] - doc.leftMargin - doc.rightMargin
    summary_table = Table(
        table_data,
        colWidths=[usable_width * 0.2, usable_width * 0.8],
        repeatRows=1,
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#d1d5db")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(summary_table)

    def _draw_page_number(canvas, _doc):
        canvas.saveState()
        canvas.setFont(base_font, 9)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(LETTER[0] - doc.rightMargin, 0.45 * inch, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_page_number, onLaterPages=_draw_page_number)


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
) -> tuple[int, int, bool]:
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
    cues_before_merge = len(clipped_cues)
    clipped_cues = merge_overlapping_cues(clipped_cues)
    cues_after_merge = len(clipped_cues)

    write_srt(clipped_cues, out_srt_path)
    vtt_written = False
    if out_vtt_path is not None:
        write_vtt(clipped_cues, out_vtt_path)
        vtt_written = True
    return cues_before_merge, cues_after_merge, vtt_written


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


def batch_texts_for_translation(
    cues: list[SubtitleCue],
    max_chars: int = 7500,
    max_cues: int = 120,
    max_minutes: int = 8,
) -> list[list[int]]:
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_chars = 0
    batch_start_ms: int | None = None
    max_span_ms = max(1, int(max_minutes)) * 60 * 1000

    for index, cue in enumerate(cues):
        text = str(cue.text or "").strip()
        if not text:
            continue

        if batch_start_ms is None:
            projected_span_ms = 0
        else:
            projected_span_ms = max(0, cue.end_ms - batch_start_ms)
        projected_chars = current_chars + len(text)
        projected_count = len(current_batch) + 1
        exceeds_limits = (
            current_batch
            and (
                projected_chars > max_chars
                or projected_count > max_cues
                or projected_span_ms > max_span_ms
            )
        )
        if exceeds_limits:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
            batch_start_ms = None

        if batch_start_ms is None:
            batch_start_ms = cue.start_ms
        current_batch.append(index)
        current_chars += len(text)

    if current_batch:
        batches.append(current_batch)
    return batches


def full_video_translate_subtitles(
    *,
    subtitle_path: Path,
    target_language: str,
    output_srt_path: Path,
    credentials_path: Path,
    job_id: str,
    max_chars: int = 7500,
    max_cues: int = 120,
    max_minutes: int = 8,
) -> dict[str, str | int | bool]:
    suffix = subtitle_path.suffix.lower()
    if suffix == ".srt":
        source_cues = parse_srt(subtitle_path)
    elif suffix == ".vtt":
        source_cues = parse_vtt(subtitle_path)
    else:
        raise ValueError("Unsupported subtitle source format.")

    if not source_cues:
        raise ValueError("No subtitle cues found in source file.")

    sample_chunks: list[str] = []
    sample_length = 0
    for cue in source_cues:
        text = str(cue.text or "").strip()
        if not text:
            continue
        if len(sample_chunks) >= 30 or sample_length >= 4000:
            break
        sample_chunks.append(text)
        sample_length += len(text)
    sample_text = " ".join(sample_chunks)[:4000]

    client = get_google_translate_client(credentials_path)
    detected_language = detect_language_text(client, sample_text) if sample_text else ""

    if detected_language == target_language:
        logger.info("[CLIPPER] skip translate: source already target (%s)", target_language)
        return {
            "status": "skipped",
            "detected_language": detected_language,
            "cues_before": len(source_cues),
            "cues_after": len(source_cues),
            "batches": 0,
            "written": False,
        }

    batches = batch_texts_for_translation(
        source_cues,
        max_chars=max_chars,
        max_cues=max_cues,
        max_minutes=max_minutes,
    )

    translated_text_by_index: dict[int, str] = {}
    for batch_indices in batches:
        texts = [source_cues[idx].text.strip() for idx in batch_indices]
        translated_batch = translate_texts(
            client,
            texts,
            target_language,
            batch_size=len(texts) or 1,
        )
        for idx, translated in zip(batch_indices, translated_batch):
            translated_text_by_index[idx] = translated.strip()

    translated_cues: list[SubtitleCue] = []
    for idx, cue in enumerate(source_cues):
        translated_cues.append(
            SubtitleCue(
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=translated_text_by_index.get(idx, ""),
            )
        )

    cues_before_trim = len(translated_cues)
    translated_cues = resolve_overlaps_by_trimming(translated_cues, epsilon_ms=1, min_duration_ms=250)
    cues_after_trim = len(translated_cues)
    max_cue_ms = max((cue.end_ms - cue.start_ms) for cue in translated_cues) if translated_cues else 0
    output_srt_path.parent.mkdir(parents=True, exist_ok=True)
    write_srt(translated_cues, output_srt_path)

    if cues_before_trim and cues_after_trim < int(cues_before_trim * 0.85):
        logger.warning(
            "[CLIPPER] fullsubs_translate low cue retention job_id=%s cues_before=%s cues_after=%s",
            job_id,
            cues_before_trim,
            cues_after_trim,
        )

    logger.info(
        "[CLIPPER] fullsubs_translate job_id=%s target=%s subtitle=%s",
        job_id,
        target_language,
        subtitle_path,
    )
    logger.info(
        "[CLIPPER] detected_source_lang=%s target=%s -> translating",
        detected_language or "unknown",
        target_language,
    )
    logger.info(
        "[CLIPPER] batching total_cues=%s batches=%s max_chars=%s max_cues=%s max_minutes=%s",
        len(source_cues),
        len(batches),
        max_chars,
        max_cues,
        max_minutes,
    )
    logger.info(
        "[CLIPPER] overlap-trim translated cues_before=%s cues_after=%s max_cue_ms=%s",
        cues_before_trim,
        cues_after_trim,
        max_cue_ms,
    )
    logger.info("[CLIPPER] wrote translated srt -> %s", output_srt_path)
    logger.info(
        "[CLIPPER] manual-check note: run full-video translated subtitles on an overlapping SRT and verify cue count remains close, overlaps_after=0, and playback no longer stacks.")

    return {
        "status": "translated",
        "detected_language": detected_language,
        "cues_before": cues_before_trim,
        "cues_after": cues_after_trim,
        "batches": len(batches),
        "written": True,
    }




def _dev_sanity_check_overlap_merge() -> None:
    sample_cues = [
        SubtitleCue(0, 2200, "a"),
        SubtitleCue(1800, 3600, "b"),
        SubtitleCue(3400, 5200, "c"),
        SubtitleCue(5100, 6900, "d"),
        SubtitleCue(6700, 8500, "e"),
        SubtitleCue(8300, 10100, "f"),
        SubtitleCue(20000, 21500, "g"),
    ]
    merged = merge_overlapping_cues(sample_cues, max_lines=2)
    max_duration = max((cue.end_ms - cue.start_ms) for cue in merged) if merged else 0
    print(f"dev_sanity total_cues={len(sample_cues)} merged_cues={len(merged)} max_merged_duration_ms={max_duration}")


if __name__ == "__main__":
    _dev_sanity_check_overlap_merge()

def _ffmpeg_subtitles_filter_arg(subs_in: Path) -> str:
    # Escape for ffmpeg filter parser: https://ffmpeg.org/ffmpeg-filters.html#Notes-on-filtergraph-escaping
    path_str = str(subs_in.resolve())
    path_str = path_str.replace("\\", "\\\\")
    path_str = path_str.replace(":", "\\:")
    path_str = path_str.replace("'", "\\\\'")
    return f"subtitles='{path_str}'"


def burn_subtitles_into_mp4(video_in: Path, subs_in: Path, video_out: Path) -> tuple[bool, str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_in),
        "-vf",
        _ffmpeg_subtitles_filter_arg(subs_in),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "24",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(video_out),
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if run.returncode == 0:
        return True, run.stderr.strip()
    return False, run.stderr.strip() or "ffmpeg burn-in failed."


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
