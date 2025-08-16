from typing import Iterable, List, Optional

def dedupe_preserve_order(items: Iterable[Optional[str]]) -> List[str]:
    """
    Remove duplicates from an iterable while preserving order.
    Ignores None values.
    """
    seen = set()
    out = []
    for x in items:
        if x is None:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out
