"""Google Scholar synchronization helpers for the portfolio app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import unescape
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
import re


GOOGLE_SCHOLAR_BASE_URL = "https://scholar.google.com"
DEFAULT_SCHOLAR_USER_ID = "jic1XEcAAAAJ"
DEFAULT_SCHOLAR_PROFILE_URL = f"{GOOGLE_SCHOLAR_BASE_URL}/citations?user={DEFAULT_SCHOLAR_USER_ID}&hl=en"
DEFAULT_SCHOLAR_IMAGE_URL = (
    f"{GOOGLE_SCHOLAR_BASE_URL}/citations?view_op=medium_photo&user={DEFAULT_SCHOLAR_USER_ID}"
)
DEFAULT_SYNC_TTL_SECONDS = 60 * 60 * 6


@dataclass
class ScholarSyncOutcome:
    updated: bool
    matched_papers: int = 0
    fetched_publications: int = 0
    error: str | None = None
    synced_at: str | None = None


def default_settings() -> dict[str, Any]:
    """Default app settings, including Scholar sync configuration."""
    return {
        "years_experience": "5",
        "client_projects": "5",
        "production_models": "8",
        "model_uptime": "99.9",
        "profile_image_url": DEFAULT_SCHOLAR_IMAGE_URL,
        "profile_image_fallback_url": "https://avatars.githubusercontent.com/u/169674746?v=4",
        "scholar_sync_enabled": True,
        "scholar_user_id": DEFAULT_SCHOLAR_USER_ID,
        "scholar_profile_url": DEFAULT_SCHOLAR_PROFILE_URL,
        "scholar_sync_ttl_seconds": DEFAULT_SYNC_TTL_SECONDS,
        "scholar_total_citations": "",
        "scholar_h_index": "",
        "scholar_i10_index": "",
        "scholar_last_synced_at": "",
        "scholar_sync_error": "",
        "scholar_affiliation": "",
        "scholar_name": "Shivogo John",
        "scholar_publications": []
    }


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def normalize_title(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def title_similarity(left: str | None, right: str | None) -> float:
    left_normalized = normalize_title(left)
    right_normalized = normalize_title(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0

    left_tokens = {token for token in left_normalized.split() if len(token) > 2}
    right_tokens = {token for token in right_normalized.split() if len(token) > 2}
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    sequence_ratio = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    containment = 1.0 if left_normalized in right_normalized or right_normalized in left_normalized else 0.0
    return max(sequence_ratio, overlap, containment * 0.9)


def _clean_html_text(raw_value: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", raw_value or "")
    return unescape(cleaned).replace("\xa0", " ").strip()


def _extract_int(raw_value: str | None) -> int:
    if not raw_value:
        return 0
    digits = re.sub(r"[^\d]", "", raw_value)
    return int(digits) if digits else 0


def _fetch_url(url: str, timeout: int = 20) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_scholar_profile(user_id: str, pagesize: int = 100) -> dict[str, Any]:
    """Fetch and parse a public Google Scholar profile."""
    query = urlencode({"hl": "en", "user": user_id, "cstart": 0, "pagesize": pagesize})
    url = f"{GOOGLE_SCHOLAR_BASE_URL}/citations?{query}"
    html = _fetch_url(url)

    if "Please show you're not a robot" in html or "The system can't perform the operation now" in html:
        raise RuntimeError("Google Scholar blocked the request with an anti-bot challenge.")

    profile = {
        "profile_url": f"{GOOGLE_SCHOLAR_BASE_URL}/citations?hl=en&user={user_id}",
        "image_url": "",
        "name": "",
        "affiliation": "",
        "citations": 0,
        "h_index": 0,
        "i10_index": 0,
        "publications": []
    }

    name_match = re.search(r'<div[^>]+id="gsc_prf_in"[^>]*>(.*?)</div>', html, re.S)
    if name_match:
        profile["name"] = _clean_html_text(name_match.group(1))

    affiliation_match = re.search(r'<div[^>]+class="gsc_prf_il"[^>]*>(.*?)</div>', html, re.S)
    if affiliation_match:
        profile["affiliation"] = _clean_html_text(affiliation_match.group(1))

    image_match = re.search(r'<img[^>]+id="gsc_prf_pup-img"[^>]+src="([^"]+)"', html, re.S)
    if image_match:
        profile["image_url"] = urljoin(GOOGLE_SCHOLAR_BASE_URL, unescape(image_match.group(1)))

    stats_match = re.search(r'<table[^>]+id="gsc_rsb_st"[^>]*>(.*?)</table>', html, re.S)
    if stats_match:
        stat_cells = re.findall(r'<td[^>]+class="gsc_rsb_std"[^>]*>(.*?)</td>', stats_match.group(1), re.S)
        if len(stat_cells) >= 5:
            profile["citations"] = _extract_int(_clean_html_text(stat_cells[0]))
            profile["h_index"] = _extract_int(_clean_html_text(stat_cells[2]))
            profile["i10_index"] = _extract_int(_clean_html_text(stat_cells[4]))

    row_pattern = re.compile(r'<tr[^>]+class="gsc_a_tr"[^>]*>(.*?)</tr>', re.S)
    title_pattern = re.compile(r'<a[^>]+class="gsc_a_at"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
    gray_pattern = re.compile(r'<div[^>]+class="gs_gray"[^>]*>(.*?)</div>', re.S)
    citation_pattern = re.compile(r'<a[^>]+class="gsc_a_ac[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
    year_pattern = re.compile(r'<span[^>]*class="gsc_a_h gsc_a_hc gs_ibl"[^>]*>(.*?)</span>|<td[^>]+class="gsc_a_y"[^>]*><span[^>]*>(.*?)</span>', re.S)

    publications = []
    for row_html in row_pattern.findall(html):
        title_match = title_pattern.search(row_html)
        if not title_match:
            continue

        gray_matches = gray_pattern.findall(row_html)
        citation_match = citation_pattern.search(row_html)
        year_match = year_pattern.search(row_html)

        citation_href = citation_match.group(1) if citation_match else ""
        citation_value = _clean_html_text(citation_match.group(2)) if citation_match else ""
        raw_year = ""
        if year_match:
            raw_year = year_match.group(1) or year_match.group(2) or ""

        publication = {
            "title": _clean_html_text(title_match.group(2)),
            "authors": _clean_html_text(gray_matches[0]) if len(gray_matches) > 0 else "",
            "venue": _clean_html_text(gray_matches[1]) if len(gray_matches) > 1 else "",
            "citations": _extract_int(citation_value),
            "year": _extract_int(_clean_html_text(raw_year)),
            "detail_url": urljoin(GOOGLE_SCHOLAR_BASE_URL, unescape(title_match.group(1))),
            "citation_url": urljoin(GOOGLE_SCHOLAR_BASE_URL, unescape(citation_href)) if citation_href else "",
        }
        publications.append(publication)

    profile["publications"] = publications
    return profile


def enrich_local_research_items(
    local_items: list[dict[str, Any]],
    scholar_publications: list[dict[str, Any]],
    synced_at: str
) -> int:
    """Attach Scholar metadata to the best matching local research items."""
    matched_count = 0
    remaining_publications = scholar_publications.copy()

    for item in local_items:
        if not item.get("id", "").startswith("paper_"):
            continue

        best_publication = None
        best_score = 0.0
        for publication in remaining_publications:
            score = title_similarity(item.get("title"), publication.get("title"))
            if score > best_score:
                best_score = score
                best_publication = publication

        if not best_publication or best_score < 0.45:
            continue

        item["scholar"] = {
            "title": best_publication.get("title", ""),
            "authors": best_publication.get("authors", ""),
            "venue": best_publication.get("venue", ""),
            "year": best_publication.get("year", 0),
            "citations": best_publication.get("citations", 0),
            "detail_url": best_publication.get("detail_url", ""),
            "citation_url": best_publication.get("citation_url", ""),
            "match_score": round(best_score, 3),
            "last_synced_at": synced_at,
        }
        item["citation_count"] = best_publication.get("citations", 0)
        item["scholar_last_synced_at"] = synced_at
        matched_count += 1
        remaining_publications.remove(best_publication)

    return matched_count


def sync_google_scholar_data(file_ops, settings: dict[str, Any], force: bool = False) -> ScholarSyncOutcome:
    """Fetch Google Scholar metadata and persist it to settings/data.json."""
    merged_settings = default_settings() | (settings or {})
    if not merged_settings.get("scholar_sync_enabled", True) and not force:
        return ScholarSyncOutcome(updated=False, error="Scholar sync is disabled in settings.")

    synced_at = iso_now()
    user_id = merged_settings.get("scholar_user_id", DEFAULT_SCHOLAR_USER_ID)

    try:
        profile = fetch_scholar_profile(user_id=user_id)
        scholar_publications = profile.get("publications", [])

        merged_settings.update(
            {
                "scholar_user_id": user_id,
                "scholar_profile_url": profile.get("profile_url") or DEFAULT_SCHOLAR_PROFILE_URL,
                "scholar_name": profile.get("name") or merged_settings.get("scholar_name", ""),
                "scholar_affiliation": profile.get("affiliation", ""),
                "scholar_total_citations": str(profile.get("citations", 0)),
                "scholar_h_index": str(profile.get("h_index", 0)),
                "scholar_i10_index": str(profile.get("i10_index", 0)),
                "scholar_last_synced_at": synced_at,
                "scholar_sync_error": "",
                "scholar_publications": scholar_publications,
            }
        )
        if profile.get("image_url"):
            merged_settings["profile_image_url"] = profile["image_url"]

        local_data = file_ops.read_data()
        matched_papers = enrich_local_research_items(local_data, scholar_publications, synced_at)

        if not file_ops.write_data(local_data):
            raise RuntimeError("Failed to persist updated research metadata to data.json.")
        if not file_ops.write_settings(merged_settings):
            raise RuntimeError("Failed to persist updated Scholar settings to settings.json.")

        return ScholarSyncOutcome(
            updated=True,
            matched_papers=matched_papers,
            fetched_publications=len(scholar_publications),
            synced_at=synced_at,
        )
    except Exception as exc:
        merged_settings["scholar_sync_error"] = str(exc)
        if force:
            merged_settings["scholar_last_synced_at"] = merged_settings.get("scholar_last_synced_at", "")
        file_ops.write_settings(merged_settings)
        return ScholarSyncOutcome(updated=False, error=str(exc), synced_at=synced_at)
