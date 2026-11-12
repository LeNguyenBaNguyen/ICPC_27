"""Helper script to create or inspect an anonymous Zenodo draft deposit and reserve a DOI.

Ensures strict compliance with ICPC Double-Anonymous Review:
- Keeps deposit in DRAFT status (never calls publish).
- Uses neutral creator metadata ("Anonymous Author(s)").
- Reserves a persistent DOI before publication.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path


def create_zenodo_draft(token, sandbox=False):
    base_url = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
    endpoint = f"{base_url}/api/deposit/depositions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    metadata = {
        "metadata": {
            "title": "Time to Detected Contradiction in Saved AI Code Explanations - Replication Package",
            "upload_type": "dataset",
            "description": "Replication package and research artifact for the empirical study 'Time to Detected Contradiction in Saved AI Code Explanations'. Contains 4,060 historical evaluations, 1,800 supplemental model observations, two-rater independent human validation labels, and replay scripts.",
            "creators": [
                {
                    "name": "Anonymous Author(s)",
                    "affiliation": "Anonymous Institution",
                }
            ],
            "access_right": "open",
            "prereserve_doi": True,
        }
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(metadata).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dep_id = data.get("id")
            reserved_doi = data.get("metadata", {}).get("prereserve_doi", {}).get("doi")
            draft_url = data.get("links", {}).get("html")
            return {
                "success": True,
                "deposit_id": dep_id,
                "reserved_doi": reserved_doi,
                "draft_url": draft_url,
                "status": "draft_unshared",
                "notes": "Deposit is in DRAFT state. Do NOT publish before camera-ready to preserve double-anonymous review.",
            }
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        return {"success": False, "error": f"HTTP {e.code}: {err_msg}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    token = os.environ.get("ZENODO_TOKEN")
    sandbox = "--sandbox" in sys.argv
    if not token:
        print("ZENODO_TOKEN environment variable is not set.")
        print("To reserve a DOI on Zenodo, run with: ZENODO_TOKEN=<your_token> python prepare_zenodo_draft.py [--sandbox]")
        sys.exit(0)

    res = create_zenodo_draft(token, sandbox=sandbox)
    print(json.dumps(res, indent=2))
