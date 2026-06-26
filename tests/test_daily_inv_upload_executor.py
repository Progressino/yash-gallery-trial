"""Daily inventory history upload must use INVENTORY_EXECUTOR, not HEAVY_EXECUTOR."""


def test_po_daily_inventory_history_uses_inventory_executor():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "backend" / "routers" / "po.py"
    text = src.read_text(encoding="utf-8")
    assert "INVENTORY_EXECUTOR.submit" in text
    assert "background_daily_inventory_upload" in text
    assert "HEAVY_EXECUTOR" not in text.split("daily-inventory-history")[1].split("@router.get")[0]
