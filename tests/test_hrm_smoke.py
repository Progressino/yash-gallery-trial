"""HRM SQLite module — departments, employees, tasks, appraisal."""

import os
import tempfile
from datetime import date

import pytest


@pytest.fixture()
def hrm_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "hrm_test.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    from backend.db import hrm_db

    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    return hrm_db


def test_hrm_department_employee_task_flow(hrm_db):
    import uuid
    hrm_db.create_department({"name": f"Accounts-{uuid.uuid4().hex[:8]}", "hod_name": "Raj"})
    depts = hrm_db.list_departments()
    assert len(depts) == 1
    dept_id = depts[0]["id"]

    code = hrm_db.create_employee({"name": "Vikash", "department_id": dept_id})
    assert code == "EMP-001"

    emps = hrm_db.list_employees()
    assert len(emps) == 1
    emp_id = emps[0]["id"]

    hrm_db.create_responsibility(
        {
            "employee_id": emp_id,
            "title": "Bill pe delivery date",
            "frequency": "Daily",
        }
    )
    resps = hrm_db.list_responsibilities(employee_id=emp_id)
    assert len(resps) == 1
    rid = resps[0]["id"]

    today = date.today().isoformat()
    assert hrm_db.mark_task(rid, today, "Done", marked_by="HOD")

    dash = hrm_db.get_hod_dashboard(dept_id, today, today)
    assert len(dash["responsibilities"]) == 1
    assert dash["responsibilities"][0]["dates"][today]["status"] == "Done"

    appraisal = hrm_db.get_appraisal(emp_id, today, today)
    assert appraisal["task_summary"]["done"] == 1

    perf = hrm_db.get_performance(dept_id, today, today)
    assert len(perf) == 1
    assert perf[0]["done_tasks"] >= 1


def test_hrm_router_import():
    from backend.routers.hrm import router

    assert router is not None
