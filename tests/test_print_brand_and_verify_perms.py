"""Action-level document verify rights vs module access."""
from backend.services.permissions import can_document_verify, may_access_erp_admin
from backend.services.print_brand import YASH_GALLERY, brand_header_html, logo_data_url


def test_can_document_verify_not_inherited_from_manager_or_production():
    assert can_document_verify("Admin") is True
    assert can_document_verify("Super Admin") is True
    assert can_document_verify("Accounts") is True
    assert can_document_verify("Finance Manager") is True
    assert can_document_verify("Audit") is True
    # Production / generic Manager must NOT auto-get Verify
    assert can_document_verify("Manager") is False
    assert can_document_verify("Executive") is False
    assert can_document_verify("Clerk") is False
    assert can_document_verify("Production") is False
    assert can_document_verify("HOD") is False


def test_manager_is_erp_admin_but_not_document_verify():
    assert may_access_erp_admin("Manager") is True
    assert can_document_verify("Manager") is False


def test_print_brand_header_embeds_yash_gallery():
    html = brand_header_html(doc_title="PURCHASE ORDER", doc_number="PO-1", department="Purchase")
    assert YASH_GALLERY["name"] in html
    assert "PURCHASE ORDER" in html
    assert "PO-1" in html
    logo = logo_data_url()
    assert logo.startswith("data:image/png;base64,") or logo == "/logo.png"
    if logo.startswith("data:"):
        assert logo in html
