import json
import os
import re
import smtplib
from email.mime.text import MIMEText
from typing import Iterable, Optional

from .models import Finding

CORRECTION_PHRASES = ("actually", "make that", "scratch that", "correction")
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty one": 21, "twenty-one": 21, "twenty two": 22,
    "twenty-two": 22, "twenty three": 23, "twenty-three": 23, "twenty four": 24,
    "twenty-four": 24, "twenty five": 25, "twenty-five": 25, "twenty six": 26,
    "twenty-six": 26, "twenty seven": 27, "twenty-seven": 27, "twenty eight": 28,
    "twenty-eight": 28, "twenty nine": 29, "twenty-nine": 29, "thirty": 30,
    "thirty one": 31, "thirty-one": 31, "thirty two": 32, "thirty-two": 32,
}

# Preferred models for Groq in order of capability and availability
PREFERRED_CHAT_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
]


# Whisper's well-known hallucinated outputs for silence/noise (which it was
# trained to transcribe as stock phrases).
_HALLUCINATION_PHRASES = (
    "thank you", "thanks for watching", "subscribe", "amara.org",
    "transcription by", "caption", "see you next", "bye", "goodbye",
    "через", "продолжение", "ottoman", "субтитры", "d made simple",
    "attention defend our", "receive messages by", "gute", "bitte",
    "danke",
    # Observed artifacts: Whisper echoing its own prompt and stock filler
    # over quiet audio. Never meaningful charting dictation.
    "clinical dental,", "clinical dental.", "clinical dental",
)

# English clinical dictation is plain ASCII. Non-ASCII output (Icelandic
# diacritics like ð/ý/í, Cyrillic, CJK, smart quotes, emoji) means the
# recognizer decoded the wrong language or hallucinated - reject it.
_ASCII_ONLY_RE = re.compile(r"^[A-Za-z0-9 ,.'\"#%()\-:;/?!&]+$")


_HALLUCINATION_REPETITION_MIN_WORDS = 4


def is_repetition_hallucination(text: str) -> bool:
    """Detects degenerate repeat loops ("Cinelli's Cinelli's", "and the and the and the")."""
    words = [w for w in re.findall(r"[a-z']+", (text or "").lower()) if w]
    if len(words) < 2:
        return False
    unique = set(words)
    if len(words) >= 2 and len(unique) == 1:
        return True
    if len(words) >= _HALLUCINATION_REPETITION_MIN_WORDS and len(unique) / len(words) < 0.4:
        return True
    return False


def sanitize_transcript(text: str) -> str:
    """Filters Whisper hallucinations before a transcript is charted.

    Returns the cleaned transcript, or '' when the audio produced nothing
    trustworthy (silence, room noise, or non-English output).
    """
    cleaned = (text or "").strip()
    # Recognizers sometimes emit typographic punctuation; normalize so valid
    # English with a curly apostrophe is not rejected by the ASCII guard.
    cleaned = cleaned.replace("\u2019", "'").replace("\u2018", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-")
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in lowered:
            return ""
    # Non-English scripts / accented gibberish are never valid English
    # clinical dictation.
    if not _ASCII_ONLY_RE.match(cleaned):
        return ""
    # Degenerate repetition loops (the model stuttering one fragment) are
    # never meaningful dictation.
    if is_repetition_hallucination(cleaned):
        return ""
    return cleaned


def normalize_surface(surface: str | None) -> str:
    if not surface:
        return "unspecified"
    value = surface.strip().lower()
    mapping = {
        "mesial": "mesial",
        "distal": "distal",
        "occlusal": "occlusal",
        "buccal": "buccal",
        "facial": "facial",
        "lingual": "lingual",
        "palatal": "lingual",
        "incisal": "incisal",
        "mod": "mod",
        "m o d": "mod",
        "mo": "mo",
        "do": "do",
    }
    return mapping.get(value, value)


def extract_tooth_number(transcript: str) -> str:
    """Find the tooth being discussed.

    An explicitly anchored reference ("tooth 14", "number two", "#9") always
    wins, so probing-depth numbers spoken afterwards can never hijack the
    result ("tooth two lingual six five six" must stay tooth 2).
    """
    phrase = transcript.lower().strip()

    # 1) Explicit digit reference: "tooth 14", "number 8", "#9"
    digit_ref = re.search(r"(?:tooth\s+number|tooth|number|no\.?|#)\s*(\d{1,2})\b", phrase)
    if digit_ref:
        tooth = int(digit_ref.group(1))
        return str(tooth) if 1 <= tooth <= 32 else ""

    # 2) Explicit word reference: "tooth fourteen", "number two"
    for token, value in sorted(NUMBER_WORDS.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?:tooth\s+number|tooth|number|no\.?|#)\s+{re.escape(token)}\b", phrase):
            return str(value)

    # 3) Fall back to the first standalone tooth-sized digit in the utterance.
    digit_match = re.search(r"\b(\d{1,2})\b", phrase)
    if digit_match:
        tooth = int(digit_match.group(1))
        if 1 <= tooth <= 32:
            return str(tooth)

    # 4) Last resort: any spoken number word (longest first so "thirty" beats "five").
    for token, value in sorted(NUMBER_WORDS.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(token)}\b", phrase):
            return str(value)

    return ""


def has_correction(transcript: str) -> bool:
    lowered = transcript.lower()
    return any(phrase in lowered for phrase in CORRECTION_PHRASES)


class DentalAI:
    """Groq-backed transcription, charting, and patient-report generation."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is required for live mode.")
        from groq import Groq

        self.client = Groq(api_key=key.strip())
        self.whisper_model = "whisper-large-v3-turbo"
        self.chat_model = model or self._detect_best_chat_model()

    def _detect_best_chat_model(self) -> str:
        try:
            available = [m.id for m in self.client.models.list().data]
            for candidate in PREFERRED_CHAT_MODELS:
                if candidate in available:
                    return candidate
            return "openai/gpt-oss-20b"
        except Exception:
            return "openai/gpt-oss-20b"

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as audio_file:
            result = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.whisper_model,
                language="en",
                temperature=0,
                response_format="verbose_json",
                # Keep this SHORT: long prompts get echoed back verbatim by
                # Whisper on quiet audio ("Clinical dental, doctor.").
                prompt="Tooth number, occlusal, buccal, lingual, caries, composite, crown, probing depth.",
            )
        text = (getattr(result, "text", "") or "").strip()
        # Whisper emits a no_speech_prob and avg_logprob per segment. Segments
        # the model itself was unsure about are hallucinations (invented words
        # over silence/noise) - drop them instead of charting garbage.
        try:
            segments = list(getattr(result, "segments", None) or [])
        except Exception:
            segments = []
        if segments:
            def seg_get(seg, key, default=None):
                # Groq verbose_json returns segments as plain dicts; some SDK
                # versions use objects. Support both.
                if isinstance(seg, dict):
                    return seg.get(key, default)
                return getattr(seg, key, default)

            kept = []
            for seg in segments:
                seg_text = (seg_get(seg, "text", "") or "").strip()
                try:
                    no_speech = float(seg_get(seg, "no_speech_prob", 0.0) or 0.0)
                    logprob = float(seg_get(seg, "avg_logprob", 0.0) or 0.0)
                except (TypeError, ValueError):
                    if seg_text:
                        kept.append(seg_text)
                    continue
                # Standard hallucination heuristic: a segment is silence-
                # invented only when the model BOTH thinks there was no speech
                # AND is unsure of what it heard. Dropping on either condition
                # alone throws away real (quiet/normalized) speech.
                if no_speech > 0.6 and logprob < -1.0:
                    continue
                if seg_text:
                    kept.append(seg_text)
            joined = " ".join(kept)
            if joined:
                text = joined
            elif segments and not any((seg_get(s, "text", "") or "").strip() for s in segments):
                # Segments existed but none carried text at all - fall back to
                # the raw full text, which the sanitizer still vets.
                pass
            else:
                # Every segment was flagged as invented-over-silence; the raw
                # combined text is garbage by definition, so discard it.
                text = ""
        return sanitize_transcript(text)

    def parse_findings(self, transcript: str, recent: Iterable[Finding] = ()) -> list[Finding]:
        # Noise gate: speech with no tooth reference and no clinical vocabulary
        # (e.g. small talk picked up by the mic) must not reach the LLM, which
        # would otherwise hallucinate a plausible-sounding finding.
        lower_text = transcript.lower()
        mentions_tooth = (
            extract_tooth_number(transcript) is not None
            or re.search(r"\b(tooth|teeth)\b", lower_text) is not None
        )
        mentions_condition = re.search(
            r"caries|cavit|decay|crown|\bcap\b|filling|composite|resin|amalgam|"
            r"pocket|probing|bleeding|\bbop\b|fracture|chip|crack|implant|missing|"
            r"extracted|root canal|endodontic|recurrent|peri-?odont|abscess|gingiv",
            lower_text,
        )
        if not mentions_tooth and not mentions_condition:
            return []

        context = json.dumps([finding.as_dict() for finding in recent])
        prompt = f"""Convert the dentist transcript into a JSON array of findings.
Schema: tooth_number, surface, finding_type, value, unit, notes, needs_review.
Use strings for tooth_number, value, and unit. needs_review must be boolean.
If this is a correction, update the matching recent finding rather than adding a duplicate.
Recent findings: {context}
Transcript: {transcript}"""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are an expert dental charting assistant. Spoken periodontal probing often gives three consecutive depths (e.g. 'four three four' means 4mm, 3mm, 4mm) and may include 'no bleeding' (bleeding absent) or 'bleeding positive'. Return {\"findings\": [{\"tooth_number\": \"14\", \"surface\": \"occlusal\", \"finding_type\": \"caries\", \"value\": \"moderate\", \"unit\": null, \"notes\": \"Active decay\", \"needs_review\": false}]} only."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            payload = json.loads(content)
            items = payload.get("findings", [])
            findings = []
            for item in items:
                # Ensure correct keys and defaults
                findings.append(Finding(
                    tooth_number=str(item.get("tooth_number", "")),
                    surface=str(item.get("surface", "unspecified")),
                    finding_type=str(item.get("finding_type", "clinical finding")),
                    value=str(item["value"]) if item.get("value") is not None else None,
                    unit=str(item["unit"]) if item.get("unit") is not None else None,
                    notes=str(item["notes"]) if item.get("notes") is not None else None,
                    needs_review=bool(item.get("needs_review", False)),
                    status=str(item.get("status", "active"))
                ))
            if not findings:
                # The LLM understood the speech but charted nothing; the
                # deterministic parser is a reliable backstop for well-formed
                # clinical phrases (e.g. spoken probing-depth triplets).
                rule_based = DemoAI().parse_findings(transcript, recent)
                if rule_based:
                    return rule_based
            return findings
        except Exception as exc:
            # Fallback to local DemoAI parser if LLM call encounters any error
            print(f"[DentalAI Warning] LLM parsing failed ({exc}), using rule-based parser fallback.")
            return DemoAI().parse_findings(transcript, recent)

    def generate_report(self, findings: Iterable[Finding]) -> str:
        payload = json.dumps([finding.as_dict() for finding in findings])
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "Write a warm, professional, plain-language dental visit summary. Explain each problem and recommend appropriate next steps in encouraging, understandable words. Do not invent diagnoses not present in findings."},
                    {"role": "user", "content": f"Approved chart findings: {payload}"},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[DentalAI Warning] LLM report failed ({exc}), using template fallback.")
            return DemoAI().generate_report(findings)


class DemoAI:
    """Deterministic local adapter for rehearsals and tests without API credentials."""

    def transcribe(self, audio_path: str) -> str:
        raise RuntimeError("Demo mode uses scripted transcript chunks; no audio file is needed.")

    def parse_findings(self, transcript: str, recent: Iterable[Finding] = ()) -> list[Finding]:
        tooth = extract_tooth_number(transcript)
        if not tooth:
            return []

        text = transcript.lower()
        surface_match = re.search(
            r"(mesial|distal|occlusal|buccal|facial|lingual|palatal|incisal|mod|mo|do)",
            transcript,
            re.IGNORECASE,
        )
        surface = normalize_surface(surface_match.group(1) if surface_match else "unspecified")
        notes: str | None = None

        # Digitize spoken number words ("four three four" -> "4 3 4") so probing
        # depth triplets are recognized even without the word "pocket"/"probing".
        digitized = re.sub(
            r"\b(" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")\b",
            lambda m: str(NUMBER_WORDS[m.group(1)]),
            text,
        )
        # Remove the tooth reference so its number is never read as a depth.
        depths_text = re.sub(r"(?:tooth\s*number|tooth|number|no\.?|#)\s*\d{1,2}", " ", digitized)
        triplet = re.search(r"(\d{1,2})\s*[-,]?\s*(\d{1,2})\s*[-,]?\s*(\d{1,2})", depths_text)
        if re.search(r"pocket|probing|bop|bleeding|millimeter|\bmm\b", text) or triplet:
            mm_match = re.search(r"(\d{1,2})\s*(?:mm|millimeters?)\b", digitized)
            depth_match = re.search(r"(?:pocket|probing)\s+(?:depth\s+)?(\d{1,2})", digitized)
            if mm_match:
                value = mm_match.group(1)
            elif depth_match:
                value = depth_match.group(1)
            elif triplet:
                value = f"{triplet.group(1)} {triplet.group(2)} {triplet.group(3)}"
            else:
                value = "present"
            finding_type = "pocket depth"
            unit = "mm"
            if re.search(r"bleeding|bop", text):
                bleeding_negated = bool(re.search(r"no bleeding|without bleeding|negative for bleeding|bleeding negative", text))
                notes = "no bleeding on probing" if bleeding_negated else "bleeding on probing positive"
            else:
                notes = None
        elif re.search(r"recurrent|margin breakdown|underneath|under the (filling|crown|restoration)", text):
            finding_type = "recurrent decay"
            value = "present"
            unit = None
        elif re.search(r"caries|decay|cavity|dental decay|tooth decay", text):
            severity = "moderate"
            if re.search(r"deep|severe|extensive|large", text):
                severity = "severe"
            elif re.search(r"early|incipient|small|mild", text):
                severity = "mild"
            finding_type = "caries"
            value = severity
            unit = None
        elif re.search(r"fracture|crack|chip|broken tooth", text):
            finding_type = "fracture"
            value = "present"
            unit = None
        elif re.search(r"crown|cap|full coverage", text):
            finding_type = "crown"
            value = "present"
            unit = None
        elif re.search(r"missing|extracted|absent", text):
            finding_type = "missing tooth"
            value = "present"
            unit = None
        elif re.search(r"composite|resin|filling|restoration|tooth colored filling", text):
            finding_type = "restoration"
            value = "present"
            unit = None
        elif re.search(r"wear|attrition|abrasion", text):
            finding_type = "tooth wear"
            value = "noted"
            unit = None
        else:
            finding_type = "clinical finding"
            value = transcript.strip()[:120]
            unit = None

        needs_review = bool(re.search(r"unclear|unsure|check|warning|conflict|maybe|possibly|likely", transcript, re.IGNORECASE))
        candidate = Finding(tooth, surface, finding_type, value, unit, notes=notes, needs_review=needs_review)

        if has_correction(transcript):
            for old in reversed(list(recent)):
                if old.tooth_number == tooth and old.status == "active":
                    old.status = "corrected"
                    candidate.notes = f"Correction to {old.finding_type} {old.display_value}"
                    break

        return [candidate]

    def generate_report(self, findings: Iterable[Finding]) -> str:
        active = [finding for finding in findings if finding.status == "active"]
        if not active:
            return "No clinical findings were recorded during this visit."
        lines = ["Dental Visit Summary", "", "Your dentist recorded the following findings:"]
        lines.extend(
            f"- Tooth #{finding.tooth_number}, {finding.surface}: {finding.finding_type} ({finding.display_value})."
            for finding in active
        )
        lines.extend(["", "Please contact your dental office with any questions about your care."])
        return "\n".join(lines)


def send_email(report: str, recipient: str, subject: str = "Your Dental Visit Summary") -> None:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "465"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    if not username or not password:
        raise RuntimeError("SMTP_USERNAME and SMTP_PASSWORD are required to send email.")
    message = MIMEText(report)
    message["Subject"] = subject
    message["From"] = username
    message["To"] = recipient
    with smtplib.SMTP_SSL(host, port) as server:
        server.login(username, password)
        server.send_message(message)
