"""Guards against O(n^2) regex backtracking (framework doc: DLP entity packs
run on every request via the Fast Scan Layer, so a quadratic pattern is a
CPU-DoS vector, not just a slow test). Empirically doubles input size and
checks time doesn't roughly quadruple — catches the shape of bug found in
email's original unbounded local-part (fixed: bounded to RFC 5321 lengths).
"""

import re
import time

import pytest

from armor.detectors.registry import default_registry

# One adversarial string per entity: repeats a character from the pattern's
# own alphabet with no valid terminator, the shape that triggers backtracking
# blowup if a quantifier is unbounded and a required literal never appears.
_PROBE_CHARS = {
    "email": "A",
    "url": "http://",
    "generic_jwt": "eyJ",
}


@pytest.mark.parametrize("entry", default_registry().entries, ids=lambda e: e.entity_id)
def test_pattern_scales_linearly_not_quadratically(entry):
    probe = _PROBE_CHARS.get(entry.entity_id, "A1")
    small = probe * 4000
    large = probe * 16000  # 4x bigger

    t_small = _timed(entry.pattern, small)
    t_large = _timed(entry.pattern, large)

    if t_small < 0.0005:  # too fast to measure a meaningful ratio
        assert t_large < 0.5, f"{entry.entity_id}: {t_large:.3f}s on 4x-larger adversarial input"
        return

    ratio = t_large / t_small
    # linear growth -> ~4x; quadratic growth -> ~16x. Allow generous slack.
    assert ratio < 8.0, f"{entry.entity_id}: time ratio {ratio:.1f} for 4x input suggests O(n^2)"


def _timed(pattern: re.Pattern, text: str) -> float:
    start = time.perf_counter()
    pattern.findall(text)
    return time.perf_counter() - start
