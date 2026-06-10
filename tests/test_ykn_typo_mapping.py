"""YKN → YK listing typo canonicalization."""

from backend.services.po_engine import canonical_oms_key


def test_ykn_typo_maps_to_yk_without_explicit_mapping():
    assert canonical_oms_key("1059YKNMUSTARD-6XL", {}) == "1059YKMUSTARD-6XL"
    assert canonical_oms_key("1112YKNBLACK-4XL", {}) == "1112YKBLACK-4XL"
    assert canonical_oms_key("146YKN148AZKBLU-7XL", {}) == "146YK148AZKBLU-7XL"


def test_ykn_typo_uses_mapping_when_present():
    m = {"1059YKNMUSTARD-6XL": "1059YKMUSTARD-6XL"}
    assert canonical_oms_key("1059YKNMUSTARD-6XL", m) == "1059YKMUSTARD-6XL"
