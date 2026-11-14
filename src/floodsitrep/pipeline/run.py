"""End-to-end glue. Owner: Antaun."""
from pathlib import Path

from floodsitrep.common.interfaces import load_evidence


def run_tile(tile_id, evidence_dir="evidence/samples", report_text=None):
    """evidence -> verified claims -> advisory. Later steps plug in as members merge."""
    evidence = load_evidence(Path(evidence_dir) / f"{tile_id}.json")
    result = {"tile_id": tile_id, "evidence_source": evidence["source"],
              "evidence": evidence, "claims": None, "advisory": None}
    try:
        from floodsitrep.nlp.checker import verify               # member 4
        if report_text:
            result["claims"] = verify(report_text, evidence)
    except ImportError:
        pass
    try:
        from floodsitrep.advisory.generate import advise          # member 5
        if result["claims"]:
            result["advisory"] = {lang: advise(result["claims"], lang) for lang in ("en", "ar")}
    except ImportError:
        pass
    return result
