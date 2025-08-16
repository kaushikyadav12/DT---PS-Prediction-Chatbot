from typing import Dict, List, Tuple

def merge_predictions(
    rule: Dict[str, List[str]],
    ml: Dict[str, List[str]]
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Merge predictions from rule-based and ML-based sources.
    Assumes ALL labels are multilabel (lists of values possible).

    Returns:
        out: merged predictions per label (unique, sorted list)
        provenance: source of each value ('rule', 'ml', or 'rule+ml')
    """
    out = {}
    provenance = {}
    all_keys = set(rule.keys()) | set(ml.keys())

    for k in all_keys:
        # Convert None or empty to empty set
        rset = set(rule.get(k) or [])
        mset = set(ml.get(k) or [])
        merged = sorted(rset | mset)  # Unique, sorted

        out[k] = merged

        # Track provenance
        prov = {}
        for v in merged:
            sources = []
            if v in rset:
                sources.append("rule")
            if v in mset:
                sources.append("ml")
            prov[v] = "+".join(sources)
        provenance[k] = prov

    return out, provenance
