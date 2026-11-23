"""Readable transcript grouping based on source text and word timing."""
from __future__ import annotations

import re


_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
    "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.",
}
_CLOSERS = '”’\"\'»)]}'


def _has_sentence_end(text: str) -> bool:
    value = text.rstrip()
    while value and value[-1] in _CLOSERS:
        value = value[:-1].rstrip()
    if not value or value[-1] not in ".!?。！？…｡؟։؛។៕":
        return False
    if value.endswith("...") or value.endswith("…"):
        return True
    if value[-1] == ".":
        token = re.search(r"(?:\b\w+\.)+$", value, re.UNICODE)
        if token and token.group(0).casefold() in _ABBREVIATIONS:
            return False
        if re.search(r"\b(?:[A-Z]\.){2,}$", value):
            return False
        # A final period after a decimal's fractional digits is still sentence punctuation.
    return True


def _join_text(left: str, right: str) -> str:
    if not left:
        return right.lstrip()
    if not right:
        return left
    if right[0].isspace() or left[-1].isspace():
        return left + right
    if right[0] in ",.!?;:%)]}。！？、，；：…”’\"'»":
        return left + right
    # Whisper word strings normally carry their own leading spaces. This fallback
    # handles test/legacy segments without inserting spaces between CJK characters.
    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return left + (" " if _is_cjk(left[-1]) != _is_cjk(right[0]) else "") + right
    return left + " " + right


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF
            or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF)


def _group_words(words: list[dict]) -> list[dict]:
    result = []
    text = ""
    start = None
    end = None
    for word in words:
        piece = str(word.get("text", ""))
        if not piece:
            continue
        word_start = float(word.get("start", 0) or 0)
        word_end = float(word.get("end", word_start) or word_start)
        pause = max(0.0, word_start - end) if end is not None else 0.0
        long_pause_boundary = (text and pause >= 1.2 and end is not None
                               and end - start >= 20.0)
        clause_pause_boundary = (text and pause >= 0.8
                                 and re.search(r"[,;:，；：]\s*$", text))
        if long_pause_boundary or clause_pause_boundary:
            result.append({"start": start, "end": end, "text": text.strip()})
            text, start, end = "", None, None
        if start is None:
            start = word_start
        text = _join_text(text, piece)
        end = word_end
        if _has_sentence_end(text):
            result.append({"start": start, "end": end, "text": text.strip()})
            text, start, end = "", None, None
    if text.strip():
        result.append({"start": start or 0, "end": end or start or 0, "text": text.strip()})
    return result


def segment_transcript(segments: list[dict], *, weak_pause: float = 0.8,
                       strong_pause: float = 1.2, soft_max_seconds: float = 20.0) -> list[dict]:
    """Combine Whisper fragments into sentences while retaining source timestamps.

    Word-level timestamps are used to split a segment that contains multiple
    sentences. Without them, a source segment is kept intact rather than assigning
    fabricated start times to sentences inside it.
    """
    has_words_for_all_segments = bool(segments) and all(segment.get("words") for segment in segments)
    word_stream = [word for segment in segments for word in (segment.get("words") or [])]
    if has_words_for_all_segments:
        sentences = _group_words(word_stream)
        source_text = "".join(str(segment.get("text", "")) for segment in segments)
        grouped_text = "".join(item["text"] for item in sentences)
        if re.sub(r"\s+", "", source_text) != re.sub(r"\s+", "", grouped_text):
            # Word alignment can occasionally omit/normalize a token. In that case
            # the segment transcript is authoritative; never sacrifice its text.
            has_words_for_all_segments = False
    if not has_words_for_all_segments:
        # Legacy/fallback mode: join source segments, retaining only timestamps
        # actually present in the input. Sentence boundaries at source boundaries
        # can be used, but text inside one source segment is never assigned a guess.
        sentences = []
        pending = None
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            start = float(segment.get("start", 0) or 0)
            end = float(segment.get("end", start) or start)
            speaker = segment.get("speaker")
            gap = 0.0
            if pending is None:
                pending = {"start": start, "end": end, "text": text}
                if speaker:
                    pending["speaker"] = speaker
            else:
                gap = max(0.0, start - pending["end"])
                boundary = _has_sentence_end(pending["text"])
                speaker_change = speaker != pending.get("speaker")
                duration = pending["end"] - pending["start"]
                clause_pause = (gap >= weak_pause and re.search(r"[,;:，；：]\s*$", pending["text"]))
                long_pause = (gap >= strong_pause and duration >= soft_max_seconds)
                if boundary or speaker_change or clause_pause or long_pause:
                    sentences.append(pending)
                    pending = {"start": start, "end": end, "text": text}
                    if speaker:
                        pending["speaker"] = speaker
                else:
                    pending["text"] = _join_text(pending["text"], text).strip()
                    pending["end"] = end
        if pending:
            sentences.append(pending)

    # When word timing is available, split at sentence marks. For long unpunctuated
    # speech, do not cut arbitrary words; source-level pause boundaries are the only
    # available safe option and are handled by fallback mode above.
    return [item for item in sentences if item["text"]]
