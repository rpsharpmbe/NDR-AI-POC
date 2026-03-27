#!/usr/bin/env python3
"""
Export every issue from Apache Fineract Jira with all fields, then write:
1) raw issue archive (NDJSON)
2) Azure AI Search-friendly flattened NDJSON
3) field catalogue and export metadata

Tested against public ASF Jira endpoints:
- /rest/api/2/field
- /rest/api/2/search

Run:
    python export_fineract_jira.py

Optional:
    python export_fineract_jira.py --project FINERACT --outdir ./jira_export
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


BASE_URL = "https://issues.apache.org/jira"
DEFAULT_PROJECT = "FINERACT"
DEFAULT_PAGE_SIZE = 100
REQUEST_TIMEOUT = 120
SLEEP_BETWEEN_REQUESTS_SECONDS = 0.35


@dataclass
class ExportStats:
    total_issues: int = 0
    pages_fetched: int = 0
    raw_file: str = ""
    ai_search_file: str = ""
    fields_file: str = ""
    metadata_file: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "fineract-jira-export/1.0",
        }
    )
    return session


def get_all_fields(session: requests.Session, base_url: str) -> List[Dict[str, Any]]:
    url = f"{base_url}/rest/api/2/field"
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def search_issues(
    session: requests.Session,
    base_url: str,
    jql: str,
    start_at: int,
    max_results: int,
) -> Dict[str, Any]:
    url = f"{base_url}/rest/api/2/search"
    params = {
        "jql": jql,
        "fields": "*all",
        "expand": "names,schema",
        "startAt": start_at,
        "maxResults": max_results,
    }
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return " ".join(str(value).split())


def extract_named_value(obj: Any, preferred_keys: Iterable[str] = ("name", "value", "displayName", "key")) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for key in preferred_keys:
            if key in obj and obj[key] is not None:
                return obj[key]
        return obj
    return obj


def normalize_field_value(value: Any) -> Any:
    """
    Produce a JSON-serializable simplified representation while keeping useful structure.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        normalized = [normalize_field_value(v) for v in value]
        return normalized

    if isinstance(value, dict):
        # Keep compact versions of common Jira objects.
        common_keys = [
            "id", "key", "name", "value", "displayName", "description",
            "self", "accountId", "emailAddress", "active", "archived",
            "released", "releaseDate", "startDate"
        ]
        reduced = {k: value[k] for k in common_keys if k in value}
        if reduced:
            # Keep children if this looks like a structured issue link / status / priority / version / component object
            if "inwardIssue" in value:
                reduced["inwardIssue"] = normalize_field_value(value["inwardIssue"])
            if "outwardIssue" in value:
                reduced["outwardIssue"] = normalize_field_value(value["outwardIssue"])
            if "statusCategory" in value:
                reduced["statusCategory"] = normalize_field_value(value["statusCategory"])
            return reduced

        # Fall back to recursive normalization
        return {k: normalize_field_value(v) for k, v in value.items()}

    return str(value)


def build_search_text(
    key: str,
    summary: str,
    description: str,
    issue_type: str,
    status: str,
    priority: str,
    labels: List[str],
    components: List[str],
    fix_versions: List[str],
    affects_versions: List[str],
    custom_field_text_parts: List[str],
) -> str:
    parts: List[str] = [
        key,
        summary,
        description,
        issue_type,
        status,
        priority,
        " ".join(labels),
        " ".join(components),
        " ".join(fix_versions),
        " ".join(affects_versions),
        " ".join(custom_field_text_parts),
    ]
    return "\n".join(p for p in parts if p and p.strip())


def flatten_issue_for_ai_search(
    issue: Dict[str, Any],
    field_name_map: Dict[str, str],
) -> Dict[str, Any]:
    fields = issue.get("fields", {}) or {}

    issue_key = issue.get("key", "")
    issue_id = issue.get("id", "")
    summary = normalize_text(fields.get("summary"))
    description = normalize_text(fields.get("description"))

    issue_type = normalize_text(extract_named_value(fields.get("issuetype")))
    status = normalize_text(extract_named_value(fields.get("status")))
    priority = normalize_text(extract_named_value(fields.get("priority")))
    resolution = normalize_text(extract_named_value(fields.get("resolution")))

    project = normalize_text(extract_named_value(fields.get("project"), preferred_keys=("key", "name")))
    reporter = normalize_text(extract_named_value(fields.get("reporter")))
    assignee = normalize_text(extract_named_value(fields.get("assignee")))
    creator = normalize_text(extract_named_value(fields.get("creator")))

    labels = [normalize_text(v) for v in (fields.get("labels") or []) if v]
    components = [
        normalize_text(extract_named_value(v))
        for v in (fields.get("components") or [])
        if v
    ]
    fix_versions = [
        normalize_text(extract_named_value(v))
        for v in (fields.get("fixVersions") or [])
        if v
    ]
    affects_versions = [
        normalize_text(extract_named_value(v))
        for v in (fields.get("versions") or [])
        if v
    ]

    created = fields.get("created")
    updated = fields.get("updated")
    resolved = fields.get("resolutiondate")
    due_date = fields.get("duedate")

    custom_fields_compact: Dict[str, Any] = {}
    custom_field_text_parts: List[str] = []

    for field_id, value in fields.items():
        if not field_id.startswith("customfield_"):
            continue

        field_name = field_name_map.get(field_id, field_id)
        normalized = normalize_field_value(value)
        custom_fields_compact[field_name] = normalized

        # Text enrichment for search
        if normalized is None:
            continue
        if isinstance(normalized, (str, int, float, bool)):
            custom_field_text_parts.append(f"{field_name}: {normalized}")
        elif isinstance(normalized, list):
            flattened_items = []
            for item in normalized:
                if isinstance(item, dict):
                    flattened_items.append(json.dumps(item, ensure_ascii=False))
                else:
                    flattened_items.append(str(item))
            if flattened_items:
                custom_field_text_parts.append(f"{field_name}: {'; '.join(flattened_items)}")
        elif isinstance(normalized, dict):
            custom_field_text_parts.append(f"{field_name}: {json.dumps(normalized, ensure_ascii=False)}")

    search_text = build_search_text(
        key=issue_key,
        summary=summary,
        description=description,
        issue_type=issue_type,
        status=status,
        priority=priority,
        labels=labels,
        components=components,
        fix_versions=fix_versions,
        affects_versions=affects_versions,
        custom_field_text_parts=custom_field_text_parts,
    )

    flattened = {
        "id": issue_key or issue_id,
        "jiraId": issue_id,
        "key": issue_key,
        "project": project,
        "issueType": issue_type,
        "summary": summary,
        "description": description,
        "status": status,
        "priority": priority,
        "resolution": resolution,
        "reporter": reporter,
        "assignee": assignee,
        "creator": creator,
        "labels": labels,
        "components": components,
        "fixVersions": fix_versions,
        "affectsVersions": affects_versions,
        "created": created,
        "updated": updated,
        "resolved": resolved,
        "dueDate": due_date,
        "customFields": custom_fields_compact,
        "searchText": search_text,
        # Keep raw fields too; useful if you later want to re-shape in Azure or downstream code.
        "rawFields": fields,
    }

    return flattened


def write_ndjson_line(handle, obj: Dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def export_project(
    base_url: str,
    project_key: str,
    outdir: Path,
    page_size: int,
) -> ExportStats:
    session = get_session()

    outdir.mkdir(parents=True, exist_ok=True)
    fields_file = outdir / "fields_catalog.json"
    metadata_file = outdir / "export_metadata.json"
    raw_file = outdir / "issues_raw.ndjson"
    ai_search_file = outdir / "issues_ai_search.ndjson"

    print(f"[{utc_now_iso()}] Fetching field catalogue...")
    fields_catalog = get_all_fields(session, base_url)
    fields_file.write_text(json.dumps(fields_catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    # Map field id -> display name
    field_name_map: Dict[str, str] = {
        field["id"]: field.get("name", field["id"])
        for field in fields_catalog
        if isinstance(field, dict) and "id" in field
    }

    jql = f"project = {project_key} ORDER BY key"

    start_at = 0
    total = None
    page_count = 0

    print(f"[{utc_now_iso()}] Exporting issues for project {project_key}...")

    with raw_file.open("w", encoding="utf-8") as raw_handle, ai_search_file.open("w", encoding="utf-8") as flat_handle:
        while True:
            page = search_issues(
                session=session,
                base_url=base_url,
                jql=jql,
                start_at=start_at,
                max_results=page_size,
            )

            issues = page.get("issues", []) or []

            if total is None:
                total = int(page.get("total", 0))
                print(f"Total issues reported by Jira: {total}")

            if not issues:
                break

            for issue in issues:
                write_ndjson_line(raw_handle, issue)

                flattened = flatten_issue_for_ai_search(issue, field_name_map)
                write_ndjson_line(flat_handle, flattened)

            page_count += 1
            start_at += len(issues)
            print(f"Fetched {start_at}/{total} issues across {page_count} page(s).")

            if start_at >= total:
                break

            time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)

    metadata = {
        "exportedAtUtc": utc_now_iso(),
        "baseUrl": base_url,
        "projectKey": project_key,
        "jql": jql,
        "pageSize": page_size,
        "totalIssues": total or 0,
        "pagesFetched": page_count,
        "files": {
            "fieldsCatalog": str(fields_file.resolve()),
            "issuesRawNdjson": str(raw_file.resolve()),
            "issuesAiSearchNdjson": str(ai_search_file.resolve()),
        },
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return ExportStats(
        total_issues=total or 0,
        pages_fetched=page_count,
        raw_file=str(raw_file.resolve()),
        ai_search_file=str(ai_search_file.resolve()),
        fields_file=str(fields_file.resolve()),
        metadata_file=str(metadata_file.resolve()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all Jira issues and fields from a public Jira project.")
    parser.add_argument("--base-url", default=BASE_URL, help="Jira base URL")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Jira project key, e.g. FINERACT")
    parser.add_argument("--outdir", default=f"./jira_export_{safe_filename(DEFAULT_PROJECT)}", help="Output directory")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Search page size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)

    try:
        stats = export_project(
            base_url=args.base_url.rstrip("/"),
            project_key=args.project,
            outdir=outdir,
            page_size=args.page_size,
        )
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        if exc.response is not None:
            try:
                print(exc.response.text, file=sys.stderr)
            except Exception:
                pass
        return 1
    except Exception as exc:
        print(f"Unhandled error: {exc}", file=sys.stderr)
        return 1

    print("\nExport complete.")
    print(f"Total issues:   {stats.total_issues}")
    print(f"Pages fetched:  {stats.pages_fetched}")
    print(f"Fields file:    {stats.fields_file}")
    print(f"Raw issues:     {stats.raw_file}")
    print(f"AI Search file: {stats.ai_search_file}")
    print(f"Metadata file:  {stats.metadata_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())