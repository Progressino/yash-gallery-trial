"""HRM issue sheet (Yash Gallery HRM.xlsx): working/free time, overlap reallocation,
rejected non-daily carry-forward and leave backup cover sync."""
from __future__ import annotations

import uuid
from datetime import date

import pytest

from backend.db import hrm_db, hrm_worktime
from backend.services.rbac import HrmScope

MON, TUE, WED, THU = "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"


class Clock:
    def __init__(self, t: str):
        self.t = t

    def __call__(self) -> str:
        return self.t


@pytest.fixture()
def env(tmp_path, monkeypatch, client):
    db_path = str(tmp_path / "hrm_sheet.db")
    monkeypatch.setenv("HRM_DB_PATH", db_path)
    monkeypatch.setattr(hrm_db, "_DB", db_path)
    hrm_db.init_db()
    monkeypatch.setattr(hrm_db, "in_task_action_window", lambda *a, **k: True)
    clock = Clock(f"{MON} 09:00:00")
    monkeypatch.setattr(hrm_db, "_now_iso", clock)
    monkeypatch.setattr(hrm_db, "today_ist", lambda: date.fromisoformat(clock.t[:10]))
    monkeypatch.setattr("backend.routers.hrm.today_ist", lambda: date.fromisoformat(clock.t[:10]))

    hrm_db.create_department({"name": f"D-{uuid.uuid4().hex[:6]}"})
    did = hrm_db.list_departments()[0]["id"]
    for n in ("Worker A", "Hod H", "Backup C"):
        hrm_db.create_employee({"name": n, "department_id": did})
    ids = {e["name"]: e["id"] for e in hrm_db.list_employees()}
    a = ids["Worker A"]
    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="self", role="Employee", user_id=1, employee_id=a, department_id=did),
    )
    return {"client": client, "a": a, "h": ids["Hod H"], "c": ids["Backup C"], "clock": clock}


def _resp(emp, **kw):
    data = {"employee_id": emp, "title": f"Resp-{uuid.uuid4().hex[:4]}", "frequency": "Daily"}
    data.update(kw)
    return hrm_db.create_responsibility(data)


def _task(emp, title):
    return hrm_db.create_one_time_task({"employee_id": emp, "title": title})


def _add_task_slot(tid, s, e, **kw):
    return hrm_worktime.add_manual_slot(
        entity_type="one_time", entity_id=tid, started_at=f"{MON} {s}", ended_at=f"{MON} {e}", actor="A", **kw
    )


def _task_slots(emp, tid):
    t = next(t for t in hrm_db.list_one_time_tasks(employee_id=emp) if t["id"] == tid)
    return t, [(s["started_at"][11:16], s["ended_at"][11:16], s.get("notes") or "") for s in t["time_slots"]]


# ── Row 2: Actual Working + Free Time ────────────────────────────────────────


def test_row2_sheet_example_free_time_deducts_breaks(env):
    a, clock = env["a"], env["clock"]
    ra, rb = _resp(a, title="Responsibility A"), _resp(a, title="Responsibility B")
    tx = _task(a, "Task X")
    hrm_worktime.office_start(a)
    clock.t = f"{MON} 18:00:00"
    for rid, s, e in ((ra, "09:00", "11:00"), (rb, "11:00", "12:00")):
        hrm_worktime.add_manual_slot(
            entity_type="responsibility", responsibility_id=rid, log_date=MON,
            started_at=f"{MON} {s}", ended_at=f"{MON} {e}", actor="A",
        )
        hrm_db.mark_task(rid, MON, "Done", "Worker A", allow_override=True)
    _add_task_slot(tx, "14:00", "15:30")
    out = hrm_worktime.office_close(a)
    assert out["office_seconds"] == 9 * 3600
    assert out["working_seconds"] == int(4.5 * 3600)
    assert out["free_seconds"] == 3 * 3600 + 45 * 60
    assert out["break_seconds"] == 45 * 60


def test_row2_working_never_exceeds_office_with_overlapping_legacy_slots(env):
    a, clock = env["a"], env["clock"]
    t1, t2 = _task(a, "Legacy 1"), _task(a, "Legacy 2")
    hrm_worktime.office_start(a)
    conn = hrm_db._connect()
    for tid, s, e in ((t1, "2026-10-04 08:00:00", f"{MON} 20:00:00"), (t2, f"{MON} 10:00:00", f"{MON} 12:00:00")):
        hrm_db._insert_closed_slot(
            conn, entity_type="one_time", entity_id=tid, employee_id=a, log_date=MON,
            started_at=s, ended_at=e, source="legacy",
        )
    conn.commit()
    conn.close()
    clock.t = f"{MON} 18:00:00"
    out = hrm_worktime.office_close(a)
    assert out["office_seconds"] == 9 * 3600
    assert out["working_seconds"] <= out["office_seconds"]
    assert out["working_seconds"] == 9 * 3600
    assert out["free_seconds"] == 0


# ── Row 3: Time slot conflict resolution & auto reallocation ─────────────────


def test_row3_partial_overlap_splits_previous_slot_and_carries_notes(env):
    a, clock = env["a"], env["clock"]
    ta, tb = _task(a, "Task A"), _task(a, "Task B")
    clock.t = f"{MON} 18:00:00"
    _add_task_slot(ta, "09:00", "10:00", notes="vendor call")
    with pytest.raises(hrm_worktime.SlotOverlap) as exc:
        _add_task_slot(tb, "09:30", "09:50")
    assert exc.value.conflicts[0]["title"] == "Task A"

    _add_task_slot(tb, "09:30", "09:50", resolve_overlap=True)
    task_a, slots_a = _task_slots(a, ta)
    assert slots_a == [("09:00", "09:30", "vendor call"), ("09:50", "10:00", "vendor call")]
    assert task_a["total_work_seconds"] == 40 * 60
    _, slots_b = _task_slots(a, tb)
    assert slots_b == [("09:30", "09:50", "")]
    assert hrm_worktime.day_time_summary(a, MON)["working_seconds"] == 60 * 60
    actions = [r["action"] for r in hrm_worktime.slot_audit(slot_id=_first_slot_id(ta))]
    assert "overlap_split" in actions


def _first_slot_id(tid):
    conn = hrm_db._connect()
    row = conn.execute(
        "SELECT id FROM hrm_time_slots WHERE entity_type='one_time' AND entity_id=? ORDER BY started_at LIMIT 1",
        (tid,),
    ).fetchone()
    conn.close()
    return int(row["id"])


def test_row3_longer_new_slot_trims_and_covering_slot_removes(env):
    a, clock = env["a"], env["clock"]
    ta, tb, tc, td = (_task(a, n) for n in ("A", "B", "C", "D"))
    clock.t = f"{MON} 18:00:00"
    _add_task_slot(ta, "09:00", "10:00")
    _add_task_slot(tb, "09:30", "11:00", resolve_overlap=True)
    assert _task_slots(a, ta)[1] == [("09:00", "09:30", "")]

    _add_task_slot(tc, "14:30", "15:00")
    _add_task_slot(td, "14:00", "16:00", resolve_overlap=True)
    task_c, slots_c = _task_slots(a, tc)
    assert slots_c == [] and int(task_c.get("active_seconds") or 0) == 0
    assert _task_slots(a, td)[1] == [("14:00", "16:00", "")]


def test_row3_running_timer_and_edit_overlap(env):
    a, clock = env["a"], env["clock"]
    ta, tb = _task(a, "Running"), _task(a, "Manual")
    assert hrm_db.start_one_time_task(ta) is True
    clock.t = f"{MON} 10:00:00"
    assert _add_task_slot(tb, "09:10", "09:20") == "overlap_running"
    assert hrm_db.pause_one_time_task(ta) is True
    out = _add_task_slot(tb, "09:10", "09:20", resolve_overlap=True)
    with pytest.raises(hrm_worktime.SlotOverlap):
        hrm_worktime.edit_time_slot(out["id"], started_at="09:10", ended_at="09:40", actor="A")


def test_row3_api_returns_409_then_resolves(env):
    c, a, clock = env["client"], env["a"], env["clock"]
    ta, tb = _task(a, "Task A"), _task(a, "Task B")
    clock.t = f"{MON} 18:00:00"
    _add_task_slot(ta, "09:00", "10:00")
    body = {"entity_type": "one_time", "task_id": tb, "started_at": f"{MON} 09:30", "ended_at": f"{MON} 09:50"}
    r = c.post("/api/hrm/time-slots", json=body)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "overlap" and detail["conflicts"][0]["title"] == "Task A"
    r = c.post("/api/hrm/time-slots", json={**body, "resolve_overlap": True})
    assert r.status_code == 200


# ── Row 4: Linked person reject → reassigned next day (non-daily) ────────────


def _log_id(emp, rid, day):
    check = hrm_db.get_employee_day_check(emp, day)
    items = check["worked_on"] + check["not_worked"] + check["other"]
    return next(i for i in items if i["responsibility_id"] == rid)["task_log_id"]


def _check_item(emp, rid, day):
    check = hrm_db.get_employee_day_check(emp, day)
    items = check["worked_on"] + check["not_worked"] + check["other"]
    return next((i for i in items if i["responsibility_id"] == rid), None)


def test_row4_weekly_reject_missed_and_reassigned_next_day(env):
    a, h, clock = env["a"], env["h"], env["clock"]
    rid = _resp(a, frequency="Weekly", schedule_weekday="Monday", linked_to_employee_id=h)
    assert hrm_db.mark_task(rid, MON, "Done", "Worker A", allow_override=True) is True
    assert hrm_db.approve_task_log(_log_id(a, rid, MON), actor="Hod H", linked_employee_id=h, action="Cancelled") is True
    assert _check_item(a, rid, MON)["status"] == "Missed"

    tue = _check_item(a, rid, TUE)
    assert tue and tue["carried_forward"] and tue["pending_since"] == MON and tue["status"] == "Pending"

    clock.t = f"{TUE} 12:00:00"
    assert hrm_db.mark_task(rid, TUE, "Done", "Worker A", allow_override=True) is True
    assert hrm_db.approve_task_log(_log_id(a, rid, TUE), actor="Hod H", linked_employee_id=h, action="Cancelled") is True
    wed = _check_item(a, rid, WED)
    assert wed and wed["pending_since"] == MON

    clock.t = f"{WED} 12:00:00"
    assert hrm_db.mark_task(rid, WED, "Done", "Worker A", allow_override=True) is True
    assert hrm_db.approve_task_log(_log_id(a, rid, WED), actor="Hod H", linked_employee_id=h, action="Approved") is True
    assert _check_item(a, rid, THU) is None
    assert _check_item(a, rid, WED)["pending_since"] == MON


def test_row4_late_reject_reappears_today(env):
    a, h, clock = env["a"], env["h"], env["clock"]
    rid = _resp(a, frequency="Weekly", schedule_weekday="Monday", linked_to_employee_id=h)
    hrm_db.mark_task(rid, MON, "Done", "Worker A", allow_override=True)
    clock.t = f"{WED} 11:00:00"
    hrm_db.approve_task_log(_log_id(a, rid, MON), actor="Hod H", linked_employee_id=h, action="Cancelled")
    assert _check_item(a, rid, WED)["pending_since"] == MON
    assert _check_item(a, rid, TUE) is None


def test_row4_daily_reject_is_missed_only(env):
    a, h = env["a"], env["h"]
    rid = _resp(a, linked_to_employee_id=h)
    hrm_db.mark_task(rid, MON, "Done", "Worker A", allow_override=True)
    hrm_db.approve_task_log(_log_id(a, rid, MON), actor="Hod H", linked_employee_id=h, action="Cancelled")
    assert _check_item(a, rid, MON)["status"] == "Missed"
    tue = _check_item(a, rid, TUE)
    assert tue and not tue["carried_forward"]


# ── Row 5: Leave → mandatory work auto-assigned to Backup Person ─────────────


def _clones(rid):
    conn = hrm_db._connect()
    rows = conn.execute(
        """SELECT reassignment_date, assignee_employee_id FROM day_reassignment_clones
           WHERE original_responsibility_id=? ORDER BY reassignment_date""",
        (rid,),
    ).fetchall()
    conn.close()
    return [(r["reassignment_date"], r["assignee_employee_id"]) for r in rows]


def test_row5_marked_mandatory_after_leave_assigns_backup(env):
    a, h, c = env["a"], env["h"], env["c"]
    rid = _resp(a, title="Fabric Approval")
    assert hrm_db.create_leave(a, MON, TUE, actor="A")["backup_assignments"] == 0

    hrm_db.update_responsibility(rid, {"mandatory": True, "backup_employee_id": h})
    assert _clones(rid) == [(MON, h), (TUE, h)]
    extra = hrm_db.get_employee_day_check(h, MON)["additional_work"]
    assert [w["responsibility_id"] for w in extra] == [rid]

    hrm_db.update_responsibility(rid, {"backup_employee_id": c})
    assert _clones(rid) == [(MON, c), (TUE, c)]

    hrm_db.update_responsibility(rid, {"mandatory": False})
    assert _clones(rid) == []


def test_row5_new_mandatory_responsibility_during_leave(env):
    a, h, clock = env["a"], env["h"], env["clock"]
    hrm_db.create_leave(a, MON, WED, actor="A")
    clock.t = f"{TUE} 10:00:00"
    rid = _resp(a, mandatory=True, backup_employee_id=h)
    assert _clones(rid) == [(TUE, h), (WED, h)]


def test_row5_backup_timer_time_and_appraisal_credit(env):
    """Leave cover: backup Start/Pause time + Done credit go to backup, not owner."""
    a, h, clock = env["a"], env["h"], env["clock"]
    rid = _resp(a, title="Fabric Approval", mandatory=True, backup_employee_id=h)
    hrm_db.create_leave(a, MON, MON, actor="A")
    assert _clones(rid) == [(MON, h)]

    hrm_worktime.office_start(h)
    assert hrm_db.start_responsibility_timer(rid, MON, allow_override=True, actor="Hod H") is True
    clock.t = f"{MON} 11:00:00"
    assert hrm_db.end_responsibility_timer(rid, MON, allow_override=True, actor="Hod H") is True
    assert hrm_db.mark_task(rid, MON, "Done", "Hod H", allow_override=True) is True

    cover = hrm_db.get_employee_day_check(h, MON)["additional_work"]
    assert len(cover) == 1 and cover[0]["status"] == "Done"
    assert cover[0]["day_work_seconds"] >= 2 * 3600 - 60
    assert cover[0].get("timer_status") in ("Completed", "Not Started") or cover[0]["day_work_seconds"] > 0

    slots = hrm_worktime.day_time_summary(h, MON)
    assert slots["working_seconds"] >= 2 * 3600 - 60
    owner_slots = hrm_worktime.day_time_summary(a, MON)
    assert owner_slots["working_seconds"] == 0

    appr_h = hrm_db.get_appraisal(h, MON, MON)
    appr_a = hrm_db.get_appraisal(a, MON, MON)
    assert appr_h["task_summary"]["done"] >= 1
    assert appr_a["task_summary"]["done"] == 0


def test_stray_approval_records_achievement_pct(env, monkeypatch):
    """Linked person must record achieved % on responsibility and one-time approval."""
    a, h, client = env["a"], env["h"], env["client"]
    rid = _resp(a, linked_to_employee_id=h)
    hrm_db.mark_task(rid, MON, "Partial", "Worker A", allow_override=True)
    log_id = _log_id(a, rid, MON)
    assert hrm_db.approve_task_log(
        log_id, actor="Hod H", linked_employee_id=h, action="Approved", achievement_pct=150,
    ) == "bad_pct"
    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="self", role="Employee", user_id=2, employee_id=h),
    )
    missing = client.post(f"/api/hrm/tasks/logs/{log_id}/approve", json={"action": "Approved"})
    assert missing.status_code == 400
    saved = client.post(
        f"/api/hrm/tasks/logs/{log_id}/approve",
        json={"action": "Cancelled", "achievement_pct": 40},
    )
    assert saved.status_code == 200
    item = _check_item(a, rid, MON)
    assert item["status"] == "Missed" and item["achievement_pct"] == 40

    tid = _task(a, "One-time review")
    assert hrm_db.start_one_time_task(tid) is True
    assert hrm_db.complete_one_time_task(tid, "submitted") is True
    assert hrm_db.approve_one_time_task(tid, "Hod H", "ok", 80) is True
    row = next(t for t in hrm_db.list_one_time_tasks(employee_id=a) if t["id"] == tid)
    assert row["status"] == "Approved" and row["achievement_pct"] == 80
