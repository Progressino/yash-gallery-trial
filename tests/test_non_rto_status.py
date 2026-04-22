"""Forward milestones like NON RTO DELIVERED must not classify as Refund (Meesho / peers)."""

from backend.services.helpers import is_non_rto_forward_milestone_status
from backend.services.meesho import _meesho_status_implies_refund


def test_non_rto_forward_milestone_detection():
    assert is_non_rto_forward_milestone_status("NON RTO DELIVERED")
    assert is_non_rto_forward_milestone_status("non-return delivered")
    assert is_non_rto_forward_milestone_status("NON-RETURN")
    assert not is_non_rto_forward_milestone_status("RTO DELIVERED")
    assert not is_non_rto_forward_milestone_status("DELIVERED")


def test_meesho_refund_classification():
    assert not _meesho_status_implies_refund("NON RTO DELIVERED")
    assert not _meesho_status_implies_refund("NON RETURN DELIVERED")
    assert _meesho_status_implies_refund("RTO")
    assert _meesho_status_implies_refund("RETURNED TO SELLER")
    assert not _meesho_status_implies_refund("DELIVERED")
