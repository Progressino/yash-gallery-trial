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


def test_one_time_task_lifecycle(hrm_db):
    import uuid

    hrm_db.create_department({"name": f"Store-{uuid.uuid4().hex[:8]}", "hod_name": "HOD"})
    dept_id = hrm_db.list_departments()[0]["id"]
    hrm_db.create_employee({"name": "Ravi", "department_id": dept_id})
    emp_id = hrm_db.list_employees()[0]["id"]

    tid = hrm_db.create_one_time_task(
        {
            "employee_id": emp_id,
            "title": "Warehouse audit",
            "description": "Complete by Friday",
            "due_date": "2026-06-06",
            "assigned_by": "HOD",
        }
    )
    tasks = hrm_db.list_one_time_tasks(employee_id=emp_id)
    assert len(tasks) == 1
    assert tasks[0]["status"] == "Pending"
    assert tasks[0]["title"] == "Warehouse audit"

    assert hrm_db.start_one_time_task(tid) is True
    started = hrm_db.list_one_time_tasks(employee_id=emp_id)[0]
    assert started["status"] == "In Progress"
    assert started["started_at"]

    assert hrm_db.complete_one_time_task(tid, "Audit checklist signed off") is True
    completed = hrm_db.list_one_time_tasks(employee_id=emp_id)[0]
    assert completed["status"] == "Done"
    assert completed["completed_at"]
    assert completed["completion_notes"] == "Audit checklist signed off"
    assert completed["duration_minutes"] >= 0

    assert hrm_db.approve_one_time_task(tid, approved_by="HOD", approval_notes="Good work") is True
    approved = hrm_db.list_one_time_tasks(employee_id=emp_id)[0]
    assert approved["status"] == "Approved"
    assert approved["approved_by"] == "HOD"
    assert approved["approved_at"]

    # Rejected tasks can be restarted
    tid2 = hrm_db.create_one_time_task({"employee_id": emp_id, "title": "Stock count"})
    hrm_db.start_one_time_task(tid2)
    hrm_db.complete_one_time_task(tid2)
    assert hrm_db.reject_one_time_task(tid2, approved_by="HOD", approval_notes="Redo section B") is True
    rejected = hrm_db.list_one_time_tasks(employee_id=emp_id, status="Rejected")[0]
    assert rejected["status"] == "Rejected"
    assert hrm_db.start_one_time_task(tid2) is True
    assert hrm_db.list_one_time_tasks(employee_id=emp_id, status="In Progress")

    appraisal = hrm_db.get_appraisal(emp_id, "2026-06-01", "2026-06-30")
    assert appraisal["one_time_summary"]["approved_on_time"] >= 1
    assert appraisal["task_summary"]["performance_pct"] is not None
