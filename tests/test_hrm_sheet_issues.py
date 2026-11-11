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
    assert out["free_seconds"] == int(4.5 * 3600)
    assert out["break_seconds"] == 0


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
    with pytest.raises(hrm_worktime.SlotOverlap) as exc:
        _add_task_slot(tb, "09:10", "09:20")
    assert exc.value.conflicts[0]["running"] is True
    out = _add_task_slot(tb, "09:10", "09:20", resolve_overlap=True)
    _, slots_a = _task_slots(a, ta)
    assert slots_a[0][:2] == ("09:00", "09:10")
    assert slots_a[1][0] == "09:20" and slots_a[1][1] == ""
    assert hrm_db.pause_one_time_task(ta) is True
    with pytest.raises(hrm_worktime.SlotOverlap):
        hrm_worktime.edit_time_slot(out["id"], started_at="09:10", ended_at="09:40", actor="A")


def test_row3_stale_open_slot_does_not_block_split(env):
    """A timer left open on an earlier day must not stop today's time split."""
    a, clock = env["a"], env["clock"]
    ta, tb = _task(a, "Old"), _task(a, "Today")
    conn = hrm_db._connect()
    hrm_db._open_slot(
        conn, entity_type="one_time", entity_id=ta, employee_id=a,
        log_date="2026-10-04", started_at="2026-10-04 11:00:00",
    )
    conn.commit()
    conn.close()
    clock.t = f"{MON} 18:00:00"
    _add_task_slot(tb, "09:00", "10:00")
    _add_task_slot(ta, "09:25", "09:37", resolve_overlap=True)
    _, old_slots = _task_slots(a, ta)
    assert ("09:25", "09:37") in [(s, e) for s, e, _ in old_slots]
    assert all(e for _, e, _ in old_slots)
    _, today = _task_slots(a, tb)
    assert today == [("09:00", "09:25", ""), ("09:37", "10:00", "")]


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


def test_dwr_other_day_does_not_auto_pause_running_timer(env):
    """Opening Daily Working Report for another day must not pause today's Active timer."""
    a, clock = env["a"], env["clock"]
    rid = _resp(a, title="Keep running")
    hrm_worktime.office_start(a)
    assert hrm_db.start_responsibility_timer(rid, MON) is True
    clock.t = f"{MON} 11:00:00"
    # Simulate Reports/DWR walking an older date for this employee.
    hrm_db.get_employee_day_check(a, "2026-10-04")
    hrm_worktime.list_dwr_report(employee_ids=[a], from_date="2026-10-04", to_date=MON)
    log = hrm_db._connect().execute(
        "SELECT started_at, paused_at, ended_at FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (rid, MON),
    ).fetchone()
    assert log["started_at"] and not log["paused_at"] and not log["ended_at"]
    open_slots = hrm_db._connect().execute(
        """SELECT COUNT(*) c FROM hrm_time_slots
           WHERE entity_type='responsibility' AND responsibility_id=? AND IFNULL(ended_at,'')=''""",
        (rid,),
    ).fetchone()["c"]
    assert open_slots == 1


def test_assign_pickers_include_every_active_employee(env, monkeypatch):
    a, h, c, client = env["a"], env["h"], env["c"], env["client"]
    did = next(e["department_id"] for e in hrm_db.list_employees() if e["id"] == a)
    hrm_db.create_department({"name": f"Other-{uuid.uuid4().hex[:6]}"})
    other_dept = next(d["id"] for d in hrm_db.list_departments() if d["name"].startswith("Other-"))
    hrm_db.create_employee({"name": "Other Dept Person", "department_id": other_dept})
    outsider = next(e["id"] for e in hrm_db.list_employees() if e["name"] == "Other Dept Person")
    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="department", role="HOD", user_id=2, employee_id=h, department_id=did),
    )
    rows = client.get("/api/hrm/employees/for-pickers").json()
    ids = {int(e["id"]) for e in rows}
    assert a in ids and h in ids and c in ids and outsider in ids


def test_hod_can_assign_one_time_task_cross_department(env, monkeypatch):
    """Assign Task pickers are org-wide; POST must accept the same roster (not silent 403)."""
    a, h, c, client = env["a"], env["h"], env["c"], env["client"]
    did = next(e["department_id"] for e in hrm_db.list_employees() if e["id"] == a)
    hrm_db.create_department({"name": f"Admin-{uuid.uuid4().hex[:6]}"})
    admin_dept = next(d["id"] for d in hrm_db.list_departments() if d["name"].startswith("Admin-"))
    hrm_db.create_employee({"name": "Harsh Admin", "department_id": admin_dept})
    harsh = next(e["id"] for e in hrm_db.list_employees() if e["name"] == "Harsh Admin")
    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="department", role="HOD", user_id=2, employee_id=h, department_id=did),
    )
    r = client.post(
        "/api/hrm/one-time-tasks",
        json={
            "employee_id": harsh,
            "title": "New design of Label and Tag",
            "description": "label design",
            "due_date": "2026-11-12",
            "priority": "Medium",
            "assigned_by": "Yogesh Kundnani",
            "linked_to_employee_id": c,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True
    tid = body.get("id")
    assert tid
    row = next(t for t in hrm_db.list_one_time_tasks(employee_id=harsh) if t["id"] == tid)
    assert row["title"] == "New design of Label and Tag"
    assert int(row["linked_to_employee_id"] or 0) == int(c)


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


# ── Rows 6–8: accountable person, approval bucket, automatic next-day reassignment ──


def test_row6_accountable_person_required(env, monkeypatch):
    a, h, client = env["a"], env["h"], env["client"]
    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="all", role="Admin", user_id=1, employee_id=h),
    )
    missing = client.post("/api/hrm/responsibilities", json={"employee_id": a, "title": "No owner", "frequency": "Daily"})
    assert missing.status_code == 400
    assert "Accountable" in missing.json()["detail"]
    same = client.post(
        "/api/hrm/responsibilities",
        json={"employee_id": a, "title": "Self", "frequency": "Daily", "linked_to_employee_id": a},
    )
    assert same.status_code == 400
    ok = client.post(
        "/api/hrm/responsibilities",
        json={"employee_id": a, "title": "Owned", "frequency": "Daily", "linked_to_employee_id": h},
    )
    assert ok.status_code == 200
    rid = next(r["id"] for r in hrm_db.list_responsibilities(employee_id=a) if r["title"] == "Owned")
    cleared = client.patch(f"/api/hrm/responsibilities/{rid}", json={"linked_to_employee_id": None})
    assert cleared.status_code == 400


def test_row7_approval_bucket_includes_one_time_and_missed_deadline(env, monkeypatch):
    a, h, client, clock = env["a"], env["h"], env["client"], env["clock"]
    rid = _resp(a, frequency="Daily", linked_to_employee_id=h, title="Daily review")
    hrm_db.mark_task(rid, MON, "Done", "Worker A", allow_override=True)
    finished = hrm_db.create_one_time_task(
        {"employee_id": a, "title": "One-time close", "due_date": MON, "linked_to_employee_id": h}
    )
    hrm_db.start_one_time_task(finished)
    assert hrm_db.complete_one_time_task(finished, "done") is True
    missed = hrm_db.create_one_time_task(
        {"employee_id": a, "title": "Overdue follow-up", "due_date": "2026-10-01", "linked_to_employee_id": h}
    )
    bucket = hrm_db.list_linked_approval_bucket(h)
    kinds = {(p["kind"], p.get("title")) for p in bucket}
    assert ("responsibility", "Daily review") in kinds
    assert ("one_time", "One-time close") in kinds
    assert ("deadline", "Overdue follow-up") in kinds
    notes = hrm_db.list_approval_notifications(h)
    assert any("Deadline 2026-10-01 reached" in n["message"] for n in notes)

    assert hrm_db.extend_one_time_deadline(missed, "2026-09-30", h) == "not_later"
    assert hrm_db.extend_one_time_deadline(missed, "2026-10-09", h) is True
    assert not any(p.get("task_id") == missed for p in hrm_db.list_linked_approval_bucket(h))
    clock.t = "2026-10-10 09:00:00"
    again = [p for p in hrm_db.list_linked_approval_bucket(h) if p.get("task_id") == missed]
    assert again and again[0]["kind"] == "deadline"

    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="self", role="Employee", user_id=2, employee_id=h),
    )
    res = client.get("/api/hrm/approvals/pending")
    assert res.status_code == 200
    titles = {p["title"] for p in res.json()["pending"]}
    assert "One-time close" in titles and "Overdue follow-up" in titles
    extended = client.post(f"/api/hrm/one-time-tasks/{missed}/extend-deadline", json={"due_date": "2026-10-12"})
    assert extended.status_code == 200


def test_row8_unfinished_nondaily_reassigns_next_working_day(env):
    a = env["a"]
    weekly = _resp(a, frequency="Weekly", schedule_weekday="Monday", title="Weekly roll")
    daily = _resp(a, frequency="Daily", title="Daily stays")
    whenever = _resp(a, frequency="Whenever Required", title="Whenever stays")
    done = _resp(a, frequency="Weekly", schedule_weekday="Monday", title="Weekly done")
    conn = hrm_db._connect()
    conn.execute(
        "UPDATE responsibilities SET created_at=? WHERE id IN (?,?,?,?)",
        (f"{MON} 08:00:00", weekly, daily, whenever, done),
    )
    conn.commit()
    conn.close()
    hrm_db.mark_task(done, MON, "Done", "Worker A", allow_override=True)

    weekly_tue = _check_item(a, weekly, TUE)
    assert weekly_tue is not None
    assert weekly_tue["carried_forward"] is True
    assert weekly_tue["pending_since"] == MON
    assert weekly_tue["status"] == "Pending"

    assert _check_item(a, daily, TUE)["carried_forward"] is False
    whenever_rows = hrm_db.get_employee_day_check(a, TUE)["whenever_required"]
    whenever_row = next(i for i in whenever_rows if i["responsibility_id"] == whenever)
    assert whenever_row["carried_forward"] is False
    finished = _check_item(a, done, TUE)
    assert finished is None or finished["carried_forward"] is False


def test_employee_cannot_add_responsibility_or_task(env):
    a, client = env["a"], env["client"]
    resp = client.post(
        "/api/hrm/responsibilities",
        json={"employee_id": a, "title": "Mine", "frequency": "Daily", "linked_to_employee_id": env["h"]},
    )
    assert resp.status_code == 403
    task = client.post("/api/hrm/one-time-tasks", json={"employee_id": a, "title": "Mine"})
    assert task.status_code == 403


def test_done_needs_working_time_and_na_goes_for_approval(env):
    a, h = env["a"], env["h"]
    rid = _resp(a, frequency="Daily", linked_to_employee_id=h, title="Needs time")
    assert hrm_db.mark_task(rid, MON, "Done", "Worker A") == "timer_required"
    assert hrm_db.mark_task(rid, MON, "N/A", "Worker A") is True
    conn = hrm_db._connect()
    log = conn.execute(
        "SELECT id, status, approval_status FROM task_logs WHERE responsibility_id=? AND log_date=?",
        (rid, MON),
    ).fetchone()
    conn.close()
    assert log["status"] == "N/A" and log["approval_status"] == "Pending"
    assert hrm_db.approve_task_log(log["id"], actor="Hod H", linked_employee_id=h, action="Cancelled") is True
    conn = hrm_db._connect()
    again = conn.execute("SELECT status, approval_status FROM task_logs WHERE id=?", (log["id"],)).fetchone()
    conn.close()
    assert again["status"] == "Pending" and again["approval_status"] == "Rejected"


def test_row9_approved_percent_is_the_responsibility_score(env):
    """Sheet row 9: Linked Person / HOD completion % becomes the responsibility score."""
    a, h, clock = env["a"], env["h"], env["clock"]
    rid = _resp(a, frequency="Daily", linked_to_employee_id=h, title="Score me")
    clock.t = f"{MON} 10:00:00"
    hrm_worktime.add_manual_slot(
        entity_type="responsibility", responsibility_id=rid, log_date=MON,
        started_at=f"{MON} 09:00", ended_at=f"{MON} 10:00", actor="A",
    )
    assert hrm_db.mark_task(rid, MON, "Done", "Worker A") is True
    conn = hrm_db._connect()
    log_id = conn.execute(
        "SELECT id FROM task_logs WHERE responsibility_id=? AND log_date=?", (rid, MON)
    ).fetchone()["id"]
    conn.close()
    assert hrm_db.approve_task_log(
        log_id, actor="Hod H", linked_employee_id=h, action="Approved", achievement_pct=80
    ) is True
    perf = hrm_db.get_performance(None, MON, MON)
    me = next(p for p in perf if p["employee_id"] == a)
    assert me["responsibility_performance_pct"] == 80.0


def test_row10_only_admin_can_change_reporting(env, monkeypatch):
    """Sheet row 10: Hierarchy reporting edits are Admin-only; HOD cannot patch."""
    a, h, client = env["a"], env["h"], env["client"]
    did = next(e["department_id"] for e in hrm_db.list_employees() if e["id"] == a)
    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="department", role="HOD", user_id=2, employee_id=h, department_id=did),
    )
    r = client.patch("/api/hrm/hierarchy/reporting", json={"employee_id": a, "reports_to_employee_id": h})
    assert r.status_code == 403
    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="all", role="Admin", user_id=1, employee_id=h, department_id=did),
    )
    r = client.patch("/api/hrm/hierarchy/reporting", json={"employee_id": a, "reports_to_employee_id": h})
    assert r.status_code == 200
    emp = next(e for e in hrm_db.list_employees() if e["id"] == a)
    assert int(emp["reports_to_employee_id"] or 0) == h


def test_row10_admin_edits_org_chart_hod_sees_full_company(env, monkeypatch):
    """Sheet row 10: Admin edits pictorial chart boxes; HOD sees the full company chart."""
    a, h, client = env["a"], env["h"], env["client"]
    did = next(e["department_id"] for e in hrm_db.list_employees() if e["id"] == a)
    hrm_db.create_department({"name": f"Other-{uuid.uuid4().hex[:6]}"})
    other_id = next(d["id"] for d in hrm_db.list_departments() if str(d["name"]).startswith("Other-"))
    hrm_db.create_employee({"name": "Other Dept Emp", "department_id": other_id})

    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="department", role="HOD", user_id=2, employee_id=h, department_id=did),
    )
    denied = client.put(
        "/api/hrm/hierarchy/org-chart",
        json={"name": "Director", "children": [{"name": "Should Fail"}]},
    )
    assert denied.status_code == 403

    hod_view = client.get("/api/hrm/hierarchy")
    assert hod_view.status_code == 200
    flat = hod_view.json()["employees_flat"]
    assert {e["name"] for e in flat} >= {"Worker A", "Hod H", "Other Dept Emp"}
    chart = hod_view.json()["org_chart"]
    assert chart["name"] == "Director"

    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="all", role="Admin", user_id=1, employee_id=h, department_id=did),
    )
    tree = {"name": "Director", "children": [{"name": "Ops Lead", "children": [{"name": "Worker A"}]}]}
    saved = client.put("/api/hrm/hierarchy/org-chart", json=tree)
    assert saved.status_code == 200
    assert saved.json()["org_chart"]["children"][0]["name"] == "Ops Lead"
    assert hrm_db.get_org_chart()["children"][0]["children"][0]["name"] == "Worker A"

    monkeypatch.setattr(
        "backend.routers.hrm._scope_from_request",
        lambda request: HrmScope(level="department", role="HOD", user_id=2, employee_id=h, department_id=did),
    )
    again = client.get("/api/hrm/hierarchy/org-chart")
    assert again.status_code == 200
    assert again.json()["children"][0]["name"] == "Ops Lead"


def test_attendance_classifies_weekoff_present_leave_and_holiday(env):
    a = env["a"]
    hrm_worktime.office_start(a)
    hrm_db.create_leave(a, TUE, TUE, actor="A")
    hrm_db.upsert_holiday(WED, "Festival")
    report = hrm_worktime.attendance_report([a], "2026-10-04", WED)
    person = report["employees"][0]
    by_day = {d["date"]: d["status"] for d in person["days"]}
    assert by_day["2026-10-04"] == "Weekoff"
    assert by_day[MON] == "Present"
    assert by_day[TUE] == "Leave"
    assert by_day[WED] == "Holiday"
    assert person["present_days"] == 1
    assert person["free_seconds"] == person["office_seconds"] - person["working_seconds"]
