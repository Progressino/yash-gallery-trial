"""Demand / Sales Order bulk line import parsing."""
from io import BytesIO

from backend.services.sales_import import (
    demand_import_template_csv,
    parse_demand_import_rows,
    parse_so_import_rows,
    so_import_template_csv,
)


def test_demand_template_has_required_columns():
    csv = demand_import_template_csv()
    assert "sku,sku_name,demand_qty" in csv.splitlines()[0]


def test_so_template_has_required_columns():
    csv = so_import_template_csv()
    header = csv.splitlines()[0]
    assert "sku" in header and "qty" in header and "rate" in header


def test_parse_demand_merges_duplicate_skus():
    lines, errors = parse_demand_import_rows(
        [
            {"sku": "A-S", "sku_name": "Style S", "demand_qty": 10},
            {"sku": "a-s", "demand_qty": 5},
            {"sku": "", "demand_qty": 1},
            {"sku": "B-M", "demand_qty": 0},
        ]
    )
    assert not any("A-S" in e and "required" in e for e in errors)
    by = {l["sku"].upper(): l for l in lines}
    assert by["A-S"]["demand_qty"] == 15
    assert by["A-S"]["sku_name"] == "Style S"
    assert "B-M" not in by
    assert any("demand_qty must be > 0" in e for e in errors)


def test_parse_so_lines():
    lines, errors = parse_so_import_rows(
        [
            {
                "sku": "X-L",
                "sku_name": "X Large",
                "qty": 20,
                "rate": 199,
                "gst_pct": 5,
                "hsn_code": "6109",
            },
            {"item_code": "X-L", "quantity": 10, "rate": 50},
        ]
    )
    assert len(lines) == 1
    assert lines[0]["qty"] == 30
    assert lines[0]["rate"] == 199
    assert lines[0]["gst_pct"] == 5


def test_sales_import_http_endpoints(client):
    td = client.get("/api/sales/demands/import-template")
    assert td.status_code == 200
    assert b"sku,sku_name,demand_qty" in td.content

    to = client.get("/api/sales/orders/import-template")
    assert to.status_code == 200
    assert b"sku," in to.content and b"qty" in to.content

    dcsv = b"sku,sku_name,demand_qty\nIMP-S,Imported S,12\nIMP-M,Imported M,8\n"
    rd = client.post(
        "/api/sales/demands/import-lines",
        files={"file": ("demand.csv", BytesIO(dcsv), "text/csv")},
    )
    assert rd.status_code == 200, rd.text
    body = rd.json()
    assert body["imported"] == 2
    assert {l["sku"] for l in body["lines"]} == {"IMP-S", "IMP-M"}

    scsv = (
        b"sku,sku_name,qty,unit,rate,hsn_code,gst_pct,merchant_code,priority,line_delivery_date,remarks\n"
        b"SO-IMP-S,Style,15,PCS,199,6109,5,,Normal,2026-10-15,\n"
    )
    rs = client.post(
        "/api/sales/orders/import-lines",
        files={"file": ("so.csv", BytesIO(scsv), "text/csv")},
    )
    assert rs.status_code == 200, rs.text
    sob = rs.json()
    assert sob["imported"] == 1
    assert sob["lines"][0]["sku"] == "SO-IMP-S"
    assert sob["lines"][0]["qty"] == 15
    assert sob["lines"][0]["rate"] == 199
