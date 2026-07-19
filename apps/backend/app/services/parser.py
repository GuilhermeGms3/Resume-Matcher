"""Document parsing service using markitdown and LLM."""

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from app.llm import complete_json, get_llm_config, get_model_name, get_safe_max_tokens
from app.prompts import PARSE_RESUME_PROMPT
from app.prompts.templates import RESUME_SCHEMA_EXAMPLE
from app.schemas import ResumeData

logger = logging.getLogger(__name__)

# Matches date ranges like "Jan 2020 - Dec 2023", "May 2021 - Present",
# "January 2020 - Current", and single dates like "Jun 2023".
_MD_DATE_RE = re.compile(
    r"(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)"
    r"\.?\s+\d{4})"
    r"(?:\s*[-–—]\s*"
    r"(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)"
    r"\.?\s+\d{4}"
    r"|Present|Current|Now|Ongoing))?",
    re.IGNORECASE,
)


def _extract_markdown_dates(markdown: str) -> list[str]:
    """Extract all month-inclusive date ranges from markdown text."""
    return _MD_DATE_RE.findall(markdown)


def restore_dates_from_markdown(
    parsed_data: dict[str, Any],
    markdown: str,
) -> dict[str, Any]:
    """Patch year-only dates in parsed data with month-inclusive dates from markdown.

    The LLM sometimes drops months during parsing (e.g. "Jun 2020 - Aug 2021"
    becomes "2020 - 2021"). This function extracts all month-inclusive dates
    from the raw markdown and replaces year-only entries where a match exists.
    """
    md_dates = _extract_markdown_dates(markdown)
    if not md_dates:
        return parsed_data

    # Build a lookup: "2020 - 2021" → "Jun 2020 - Aug 2021"
    year_to_full: dict[str, str] = {}
    year_only_re = re.compile(r"\d{4}")
    for md_date in md_dates:
        years_in_date = year_only_re.findall(md_date)
        if years_in_date:
            # Create year-only key like "2020 - 2021" or "2023"
            year_key = " - ".join(years_in_date)
            # Keep the first (most specific) match
            if year_key not in year_to_full:
                # Normalize separators
                normalized = re.sub(r"\s*[-–—]\s*", " - ", md_date.strip())
                year_to_full[year_key] = normalized

    if not year_to_full:
        return parsed_data

    patched = 0
    for section_key in ("workExperience", "education", "personalProjects"):
        for entry in parsed_data.get(section_key, []):
            if not isinstance(entry, dict):
                continue
            years = entry.get("years", "")
            if not isinstance(years, str) or not years:
                continue
            # Skip if already has months
            if re.search(
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
                years,
                re.IGNORECASE,
            ):
                continue
            # Try to find a matching month-inclusive date
            if years in year_to_full:
                entry["years"] = year_to_full[years]
                patched += 1

    # Custom sections
    custom = parsed_data.get("customSections", {})
    if isinstance(custom, dict):
        for section in custom.values():
            if not isinstance(section, dict) or section.get("sectionType") != "itemList":
                continue
            for item in section.get("items", []):
                if not isinstance(item, dict):
                    continue
                years = item.get("years", "")
                if not isinstance(years, str) or not years:
                    continue
                if re.search(
                    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
                    years,
                    re.IGNORECASE,
                ):
                    continue
                if years in year_to_full:
                    item["years"] = year_to_full[years]
                    patched += 1

    if patched:
        logger.info("Restored months in %d date fields from raw markdown", patched)

    return parsed_data


def normalize_llm_resume_data(parsed_data: dict[str, Any]) -> dict[str, Any]:
    """Coerce common local-LLM schema mistakes before Pydantic validation."""
    data = dict(parsed_data)

    personal = data.get("personalInfo")
    if not isinstance(personal, dict):
        personal = {}
    for key in ("name", "title", "email", "phone", "location"):
        if personal.get(key) is None:
            personal[key] = ""
    data["personalInfo"] = personal

    for section, string_fields, list_fields in (
        ("workExperience", ("title", "company", "years"), ("description",)),
        ("education", ("institution", "degree", "years"), ()),
        ("personalProjects", ("name", "role", "years"), ("description",)),
    ):
        items = data.get(section)
        if items is None:
            data[section] = []
            continue
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            item.setdefault("id", index)
            for key in string_fields:
                if item.get(key) is None:
                    item[key] = ""
            for key in list_fields:
                if item.get(key) is None:
                    item[key] = []

    additional = data.get("additional")
    if not isinstance(additional, dict):
        additional = {}
    for key in ("technicalSkills", "languages", "certificationsTraining", "awards"):
        if additional.get(key) is None:
            additional[key] = []
    data["additional"] = additional

    return data


async def parse_document(content: bytes, filename: str) -> str:
    """Convert PDF/DOCX to Markdown using markitdown.

    Args:
        content: Raw file bytes
        filename: Original filename for extension detection

    Returns:
        Markdown text content
    """
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".md", ".json"}:
        return content.decode("utf-8-sig", errors="replace")

    # Write to temp file for markitdown
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        md = MarkItDown()
        result = md.convert(str(tmp_path))
        return result.text_content
    finally:
        tmp_path.unlink(missing_ok=True)


async def parse_resume_to_json(markdown_text: str) -> dict[str, Any]:
    """Parse resume markdown to structured JSON using LLM.

    After LLM parsing, patches any year-only dates with month-inclusive
    dates extracted from the raw markdown. This ensures months are never
    lost regardless of LLM behavior.

    Args:
        markdown_text: Resume content in markdown format

    Returns:
        Structured resume data matching ResumeData schema
    """
    prompt = PARSE_RESUME_PROMPT.format(
        schema=RESUME_SCHEMA_EXAMPLE,
        resume_text=markdown_text,
    )

    config = get_llm_config()
    model_name = get_model_name(config)
    result = await complete_json(
        prompt=prompt,
        system_prompt="You are a JSON extraction engine. Output only valid JSON, no explanations.",
        max_tokens=get_safe_max_tokens(model_name),
        retries=3,
    )

    result = normalize_llm_resume_data(result)

    # Patch dates: restore months the LLM may have dropped
    result = restore_dates_from_markdown(result, markdown_text)

    # Validate against schema
    validated = ResumeData.model_validate(result)
    return validated.model_dump()


def parse_resume_locally(markdown_text: str) -> dict[str, Any]:
    """Build a basic structured resume without calling an LLM.

    This is a reliability fallback for local models that fail to return valid
    JSON. It keeps uploads usable; AI tailoring can still run later.
    """
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
    text = "\n".join(lines)

    email_match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
    phone_match = None
    for line in lines[:20]:
        if re.fullmatch(r"\d{4}\s*[-\u2013\u2014]\s*(?:\d{4}|Em andamento|Atual|Present)", line, re.IGNORECASE):
            continue
        candidate = re.search(r"(?:\+?\d[\d ().-]{7,}\d)", line)
        if candidate and len(re.findall(r"\d", candidate.group(0))) >= 8:
            phone_match = candidate
            break

    name_parts: list[str] = []
    for line in lines[:8]:
        if "@" in line or ":" in line or "/" in line:
            continue
        if re.search(r"\d", line):
            continue
        if len(line.split()) <= 4:
            name_parts.append(line.title())
        if len(" ".join(name_parts).split()) >= 2:
            break

    title = ""
    for line in lines:
        if line.lower().startswith(("area:", "area de atuacao:", "area de atuação:", "área:")):
            title = line.split(":", 1)[1].strip()
            break

    location = ""
    for line in lines[:15]:
        if re.search(r"\b[A-Z]{2}\b", line) and "," in line:
            location = line
            break

    skill_candidates: list[str] = []
    for line in lines:
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        if any(
            word in label.lower()
            for word in ("skill", "compet", "ferrament", "infra", "rede", "observ", "autom")
        ):
            for item in re.split(r",|;|/|\u2022", value):
                item = item.strip(" .")
                if item and len(item) <= 40:
                    skill_candidates.append(item)

    seen: set[str] = set()
    skills = []
    for skill in skill_candidates:
        key = skill.casefold()
        if key not in seen:
            seen.add(key)
            skills.append(skill)

    bullets = [
        re.sub(r"^[\u2022\-*]\s*", "", line).strip()
        for line in lines
        if line.startswith(("\u2022", "-", "*"))
    ]

    fallback = ResumeData(
        personalInfo={
            "name": " ".join(name_parts),
            "title": title,
            "email": email_match.group(0) if email_match else "",
            "phone": phone_match.group(0) if phone_match else "",
            "location": location,
        },
        summary="\n".join(lines[:12])[:1200],
        workExperience=[
            {
                "id": 1,
                "title": title or "Experiencia profissional",
                "company": "",
                "years": "",
                "description": bullets[:8],
            }
        ]
        if bullets
        else [],
        additional={"technicalSkills": skills},
        customSections={
            "raw_resume_text": {
                "sectionType": "text",
                "text": markdown_text[:8000],
            }
        },
    )
    return fallback.model_dump()
