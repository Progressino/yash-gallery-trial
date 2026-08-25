"""HRM Module Router — task tracking, issues, appraisal (RBAC-scoped)."""
import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from typing import Optional

from ..db.hrm_db import (
    list_departments,
    create_department,
    update_department,
    list_employees,
    create_employee,
    update_employee,
    delete_employee,
    import_responsibilities,
    import_one_time_tasks,
    list_responsibilities,
    create_responsibility,
    update_responsibility,
    delete_responsibility,
    mark_task,
    approve_task_log,
    reassign_mandatory_for_day,
    mark_reassignment_clone,
    process_auto_closures_ist,
    get_task_logs,
    list_issues,
    create_issue,
    resolve_issue,
    update_issue,
    update_issue_status,
    get_issue_raw,
    add_issue_comment,
    list_issue_comments,
    add_issue_attachment,
    list_issue_attachments,
    list_issue_history,
    log_voice_transcription,
    list_issue_notifications,
    delete_issue,
    get_hod_dashboard,
    get_appraisal,
    get_employee_day_check,
    list_dwr_rows,
    mark_unmarked_daily_as_missed,
    get_performance,
    employee_department_id,
    list_one_time_tasks,
    create_one_time_task,
    update_one_time_task,
    cancel_one_time_task,
    get_one_time_task_owner,
    start_one_time_task,
    complete_one_time_task,
    approve_one_time_task,
    reject_one_time_task,
    get_dashboard_stats,
    get_org_hierarchy,
    get_task_based_report,
    find_employees_by_name_prefix,
    set_manual_task_duration,
    start_responsibility_timer,
    end_responsibility_timer,
    pause_responsibility_timer,
    resume_responsibility_timer,
    set_responsibility_manual_time,
    get_responsibility_timer_detail,
    FREQUENCIES,
    PRIORITIES,
    TIME_PERIODS,
    WEEKDAYS,
    MONTH_NAMES,
    PERFORMANCE_CUTOVER_DATE,
)
from ..db.users_db import get_user_auth_profile, search_active_users, get_user_by_id
from ..services.rbac import (
    build_hrm_scope,
    hrm_scope_filters,
    assert_department_in_scope,
    assert_employee_in_scope,
    assert_hrm_write_org,
    assert_hrm_hod_or_admin,
    assert_hrm_admin_mutate_records,
    assert_hrm_delete_allowed,
    assert_can_view_employee_list,
    assert_responsibility_in_scope,
    HrmScope,
)

router = APIRouter()


def _scope_from_request(request: Request) -> HrmScope:
    payload = getattr(request.state, "auth", None) or {}
    username = payload.get("sub")
    profile = get_user_auth_profile(username) if username else None
    role = (profile or {}).get("role_name") or payload.get("role")
    return build_hrm_scope(profile, role=role)


def _recorder_from_request(request: Request) -> tuple[int | None, str]:
    """Immutable Recorded By = current logged-in user (id + display name)."""
    payload = getattr(request.state, "auth", None) or {}
    username = payload.get("sub")
    profile = get_user_auth_profile(username) if username else None
    if profile:
        uid = profile.get("id")
        try:
            uid = int(uid) if uid is not None else None
        except (TypeError, ValueError):
            uid = None
        name = (
            (profile.get("full_name") or "").strip()
            or (profile.get("username") or username or "").strip()
            or "Unknown"
        )
        return uid, name
    name = (payload.get("full_name") or payload.get("sub") or "Unknown").strip()
    return None, name


def _resolve_subject_from_user(subject_user_id: int | None, employee_id: int | None) -> dict:
    """Map ERP user selection to HR employee_id + display names for issue create/update."""
    result: dict = {
        "employee_id": employee_id,
        "subject_user_id": subject_user_id,
        "subject_user_name": "",
        "department_id": None,
        "designation": "",
        "caused_by_user_id": None,
        "caused_by_user_name": "",
        "caused_by_employee_id": None,
    }
    if subject_user_id:
        u = get_user_by_id(int(subject_user_id))
        if not u or not int(u.get("active") or 0):
            raise HTTPException(400, "Employee user is inactive or not found")
        result["subject_user_id"] = int(u["id"])
        result["subject_user_name"] = (u.get("full_name") or u.get("username") or "").strip()
        if u.get("employee_id"):
            result["employee_id"] = int(u["employee_id"])
        if u.get("hrm_department_id"):
            result["department_id"] = int(u["hrm_department_id"])
        result["designation"] = (u.get("role_name") or u.get("department") or "").strip()
    if not result["employee_id"]:
        raise HTTPException(
            400,
            "Selected user is not linked to an HR employee profile. "
            "Link employee_id on the ERP user (Admin) or pick a linked user / HR employee.",
        )
    return result


def _enforce_list_employee_scope(scope: HrmScope, employee_id: int | None) -> None:
    if employee_id is not None and int(employee_id) > 0:
        assert_employee_in_scope(scope, employee_id)


class DepartmentIn(BaseModel):
    name: str
    description: Optional[str] = ""
    hod_name: Optional[str] = ""


class DepartmentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    hod_name: Optional[str] = None


class EmployeeIn(BaseModel):
    name: str
    department_id: Optional[int] = None
    designation: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
    join_date: Optional[str] = ""
    emp_code: Optional[str] = ""
    reports_to_employee_id: Optional[int] = None


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    department_id: Optional[int] = None
    designation: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    join_date: Optional[str] = None
    status: Optional[str] = None
    emp_code: Optional[str] = None
    reports_to_employee_id: Optional[int] = None


class ResponsibilityIn(BaseModel):
    employee_id: int
    department_id: Optional[int] = None
    title: str
    description: Optional[str] = ""
    frequency: Optional[str] = "Daily"
    category: Optional[str] = "General"
    added_by: Optional[str] = ""
    priority: Optional[str] = "Medium"
    mandatory: Optional[bool] = False
    schedule_weekday: Optional[str] = ""
    schedule_month_day: Optional[int] = 0
    schedule_month: Optional[int] = 0
    time_period: Optional[str] = ""
    linked_to_employee_id: Optional[int] = None
    backup_employee_id: int
    backup_allocation_value: float
    backup_allocation_unit: Optional[str] = "days"


class ResponsibilityUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    frequency: Optional[str] = None
    category: Optional[str] = None
    employee_id: Optional[int] = None
    active: Optional[int] = None
    added_by: Optional[str] = None
    priority: Optional[str] = None
    mandatory: Optional[bool] = None
    schedule_weekday: Optional[str] = None
    schedule_month_day: Optional[int] = None
    schedule_month: Optional[int] = None
    time_period: Optional[str] = None
    linked_to_employee_id: Optional[int] = None
    backup_employee_id: Optional[int] = None
    backup_allocation_value: Optional[float] = None
    backup_allocation_unit: Optional[str] = None


class TaskApproveIn(BaseModel):
    action: str = "Approved"  # Approved | Cancelled
    notes: Optional[str] = ""
    approved_by: Optional[str] = ""


class ReassignDayIn(BaseModel):
    original_responsibility_id: int
    to_employee_id: int
    reassignment_date: str
    assigned_by: Optional[str] = ""


class ReassignMarkIn(BaseModel):
    status: str = "Done"
    remarks: Optional[str] = ""
    marked_by: Optional[str] = ""


class ManualDurationIn(BaseModel):
    duration: str  # minutes or HH:MM


class HierarchyDeptIn(BaseModel):
    department_id: int
    parent_department_id: Optional[int] = None
    hod_name: Optional[str] = None


class HierarchyReportIn(BaseModel):
    employee_id: int
    reports_to_employee_id: Optional[int] = None


class TaskMarkIn(BaseModel):
    responsibility_id: int
    log_date: str
    status: str = "Done"
    marked_by: Optional[str] = ""
    remarks: Optional[str] = ""
    blocker_employee_id: Optional[int] = None
    blocker_reason: Optional[str] = ""


class ResponsibilityTimerIn(BaseModel):
    log_date: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


class IssueIn(BaseModel):
    employee_id: Optional[int] = None
    subject_user_id: Optional[int] = None
    department_id: Optional[int] = None
    issue_date: Optional[str] = None
    issue_type: Optional[str] = "General"
    severity: Optional[str] = "Minor"
    title: str
    description: Optional[str] = ""
    # recorded_by ignored from client — always set from auth
    recorded_by: Optional[str] = None
    caused_by_employee_id: Optional[int] = None
    caused_by_user_id: Optional[int] = None
    caused_by_dept_id: Optional[int] = None
    status: Optional[str] = "Open"
    designation: Optional[str] = None


class IssueUpdateIn(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    employee_id: Optional[int] = None
    subject_user_id: Optional[int] = None
    caused_by_employee_id: Optional[int] = None
    caused_by_user_id: Optional[int] = None
    department_id: Optional[int] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    status: Optional[str] = None
    resolution: Optional[str] = None
    issue_date: Optional[str] = None
    designation: Optional[str] = None
    # must be ignored even if sent
    recorded_by: Optional[str] = None
    recorded_by_user_id: Optional[int] = None


class IssueResolveIn(BaseModel):
    resolution: str = ""


class IssueStatusIn(BaseModel):
    status: str
    resolution: Optional[str] = ""


class IssueCommentIn(BaseModel):
    comment_text: str


class IssueAttachmentIn(BaseModel):
    file_name: str
    file_url: Optional[str] = ""
    content_type: Optional[str] = ""
    file_size: Optional[int] = 0


class VoiceTranscriptionIn(BaseModel):
    transcript: str = ""
    target_field: str = "description"
    status: str = "success"
    error_message: str = ""
    issue_id: Optional[int] = None


class OneTimeTaskIn(BaseModel):
    employee_id: int
    department_id: Optional[int] = None
    title: str
    description: Optional[str] = ""
    due_date: Optional[str] = ""
    assigned_by: Optional[str] = ""
    priority: Optional[str] = "Medium"
    linked_to_employee_id: Optional[int] = None
    backup_employee_id: int
    backup_allocation_value: float
    backup_allocation_unit: Optional[str] = "days"


class OneTimeTaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[str] = None
    employee_id: Optional[int] = None
    assigned_by: Optional[str] = None
    priority: Optional[str] = None
    duration_minutes: Optional[int] = None
    manual_duration_minutes: Optional[int] = None
    linked_to_employee_id: Optional[int] = None
    backup_employee_id: Optional[int] = None
    backup_allocation_value: Optional[float] = None
    backup_allocation_unit: Optional[str] = None


class OneTimeTaskNotesIn(BaseModel):
    notes: Optional[str] = ""


class OneTimeTaskApprovalIn(BaseModel):
    approved_by: Optional[str] = ""
    notes: Optional[str] = ""


@router.get("/scope")
def get_hrm_scope(request: Request):
    """Current user's HRM visibility (for UI defaults)."""
    scope = _scope_from_request(request)
    return {
        "level": scope.level,
        "role": scope.role,
        "employee_id": scope.employee_id,
        "department_id": scope.department_id,
        "can_manage_org": scope.can_manage_org,
        "can_edit_assignments": scope.can_edit_assignments,
        "can_mutate_assignment_records": scope.can_mutate_assignment_records,
        "can_view_employee_list": scope.can_view_employee_list,
        "can_delete_hrm_records": scope.can_delete_hrm_records,
        "can_use_employee_check": scope.can_use_employee_check,
        "can_view_dashboard": scope.can_view_dashboard,
        "can_create_issues": scope.can_create_issues,
        "can_edit_issues": scope.can_edit_issues,
        "can_change_issue_status": scope.can_change_issue_status,
        "can_delete_issues": scope.can_delete_issues,
    }


@router.get("/meta")
def get_hrm_meta():
    """Dropdown options for forms (frequencies, priorities, etc.)."""
    return {
        "frequencies": list(FREQUENCIES),
        "priorities": list(PRIORITIES),
        "time_periods": list(TIME_PERIODS),
        "weekdays": list(WEEKDAYS),
        "months": [{"value": i + 1, "label": n} for i, n in enumerate(MONTH_NAMES)],
        "performance_cutover": PERFORMANCE_CUTOVER_DATE,
    }


@router.get("/departments")
def get_departments(request: Request):
    scope = _scope_from_request(request)
    dept_f, _ = hrm_scope_filters(scope)
    if dept_f == -1:
        return []
    return list_departments(dept_f)


@router.post("/departments")
def post_department(body: DepartmentIn, request: Request):
    assert_hrm_write_org(_scope_from_request(request))
    create_department(body.model_dump())
    return {"ok": True}


@router.patch("/departments/{did}")
def patch_department(did: int, body: DepartmentUpdate, request: Request):
    assert_hrm_write_org(_scope_from_request(request))
    update_department(did, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}


@router.get("/employees")
def get_employees(
    request: Request,
    department_id: Optional[int] = None,
    status: str = "Active",
):
    scope = _scope_from_request(request)
    dept_f, emp_f = hrm_scope_filters(scope, department_id=department_id)
    if emp_f == -1 or dept_f == -1:
        return []
    # Full org roster is Admin / Super Admin only; others get scoped rows for dropdowns.
    if not scope.can_view_employee_list and dept_f is None and emp_f is None:
        if scope.level == "department" and scope.department_id is not None:
            dept_f = scope.department_id
        elif scope.level == "self" and scope.employee_id is not None:
            emp_f = scope.employee_id
        else:
            raise HTTPException(403, "Only Admin can view the employees list")
    if emp_f is not None:
        return list_employees(dept_f, status, employee_id=emp_f)
    return list_employees(dept_f, status)


@router.post("/employees")
def post_employee(body: EmployeeIn, request: Request):
    scope = _scope_from_request(request)
    if scope.can_manage_org:
        pass
    elif scope.is_hod and scope.department_id is not None:
        if body.department_id is None or int(body.department_id) != int(scope.department_id):
            raise HTTPException(403, "HOD can only add employees to their department")
    else:
        raise HTTPException(403, "Not allowed to create employees")
    try:
        code = create_employee(body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "emp_code": code}


@router.patch("/employees/{eid}")
def patch_employee(eid: int, body: EmployeeUpdate, request: Request):
    scope = _scope_from_request(request)
    assert_employee_in_scope(scope, eid)
    if not scope.can_manage_org and not scope.is_hod:
        raise HTTPException(403, "Not allowed to edit employees")
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    if scope.is_hod and "department_id" in data and scope.department_id is not None:
        if int(data["department_id"]) != int(scope.department_id):
            raise HTTPException(403, "Cannot move employee out of your department")
    if "emp_code" in data and not scope.can_manage_org:
        raise HTTPException(403, "Only Admin can change Employee ID")
    try:
        update_employee(eid, data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@router.get("/employees/autocomplete")
def employee_autocomplete(request: Request, q: str = ""):
    scope = _scope_from_request(request)
    if not q.strip():
        return []
    rows = find_employees_by_name_prefix(q, limit=20)
    if scope.level == "department" and scope.department_id is not None:
        rows = [r for r in rows if r.get("department_id") == scope.department_id]
    elif scope.level == "self" and scope.employee_id is not None:
        rows = [r for r in rows if r.get("id") == scope.employee_id]
    return rows


@router.delete("/employees/{eid}")
def del_employee(eid: int, request: Request):
    scope = _scope_from_request(request)
    if not scope.can_view_employee_list:
        raise HTTPException(403, "Only Admin can delete employees")
    assert_employee_in_scope(scope, eid)
    if not delete_employee(eid):
        raise HTTPException(404, "Employee not found")
    return {"ok": True}


def _parse_import_rows(content: bytes, filename: str) -> list[dict]:
    name = (filename or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise HTTPException(400, "Upload .csv, .xlsx, or .xls")
    if df.empty:
        return []
    return df.fillna("").to_dict(orient="records")


@router.post("/import/responsibilities")
async def post_import_responsibilities(request: Request, file: UploadFile = File(...)):
    scope = _scope_from_request(request)
    if not scope.can_edit_assignments:
        raise HTTPException(403, "Not allowed to import responsibilities")
    content = await file.read()
    rows = _parse_import_rows(content, file.filename or "")
    result = import_responsibilities(rows)
    return {"ok": True, **result}


@router.post("/import/one-time-tasks")
async def post_import_one_time_tasks(request: Request, file: UploadFile = File(...)):
    scope = _scope_from_request(request)
    if scope.is_employee:
        raise HTTPException(403, "Not allowed to import tasks")
    content = await file.read()
    rows = _parse_import_rows(content, file.filename or "")
    result = import_one_time_tasks(rows)
    return {"ok": True, **result}


@router.get("/responsibilities")
def get_responsibilities(
    request: Request,
    employee_id: Optional[int] = None,
    department_id: Optional[int] = None,
):
    scope = _scope_from_request(request)
    dept_f, emp_f = hrm_scope_filters(scope, department_id=department_id, employee_id=employee_id)
    if emp_f == -1 or dept_f == -1:
        return []
    _enforce_list_employee_scope(scope, emp_f)
    return list_responsibilities(emp_f, dept_f)


@router.post("/responsibilities")
def post_responsibility(body: ResponsibilityIn, request: Request):
    scope = _scope_from_request(request)
    assert_employee_in_scope(scope, body.employee_id)
    assert_employee_in_scope(scope, body.backup_employee_id)
    if scope.is_employee and scope.employee_id != body.employee_id:
        raise HTTPException(403, "Cannot assign responsibilities for other employees")
    if int(body.backup_employee_id) == int(body.employee_id):
        raise HTTPException(400, "Backup person must be different from the assigned employee")
    if float(body.backup_allocation_value or 0) <= 0:
        raise HTTPException(400, "Backup allocation duration must be greater than zero")
    data = body.model_dump()
    data["require_backup"] = True
    if not (data.get("added_by") or "").strip():
        _, name = _recorder_from_request(request)
        data["added_by"] = name
    try:
        create_responsibility(data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@router.patch("/responsibilities/{rid}")
def patch_responsibility(rid: int, body: ResponsibilityUpdate, request: Request):
    scope = _scope_from_request(request)
    assert_hrm_admin_mutate_records(scope)
    from ..db.hrm_db import get_responsibility_owner

    owner = get_responsibility_owner(rid)
    if owner is None:
        raise HTTPException(404, "Responsibility not found")
    assert_employee_in_scope(scope, owner)
    if body.employee_id is not None:
        assert_employee_in_scope(scope, body.employee_id)
    try:
        # exclude_unset: only fields the client sent; keep False/0/"" so mandatory
        # and schedule clears persist (do not drop falsy values).
        update_responsibility(rid, body.model_dump(exclude_unset=True))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@router.delete("/responsibilities/{rid}")
def del_responsibility(rid: int, request: Request):
    scope = _scope_from_request(request)
    assert_hrm_delete_allowed(scope)
    from ..db.hrm_db import get_responsibility_owner

    owner = get_responsibility_owner(rid)
    if owner is None:
        raise HTTPException(404, "Responsibility not found")
    assert_employee_in_scope(scope, owner)
    delete_responsibility(rid)
    return {"ok": True}


@router.post("/tasks/mark")
def post_mark_task(body: TaskMarkIn, request: Request):
    scope = _scope_from_request(request)
    assert_responsibility_in_scope(scope, body.responsibility_id)
    if body.blocker_employee_id:
        assert_employee_in_scope(scope, body.blocker_employee_id)
    ok = mark_task(
        body.responsibility_id,
        body.log_date,
        body.status,
        body.marked_by or "",
        body.remarks or "",
        body.blocker_employee_id,
        body.blocker_reason or "",
        allow_override=scope.can_edit_assignments,
    )
    if ok is True:
        return {"ok": True}
    if ok == "locked":
        raise HTTPException(409, "Status already set and cannot be changed")
    if ok == "window_closed":
        raise HTTPException(
            409,
            "Status edit window closed (editable task date through next 2 IST days)",
        )
    if ok == "invalid_status":
        raise HTTPException(400, "Invalid status")
    raise HTTPException(404, "Responsibility not found")


def _timer_http_result(ok):
    if ok is True:
        return {"ok": True}
    if ok == "window_closed":
        raise HTTPException(
            409,
            "Time can be recorded on the task date through the next 2 IST days",
        )
    if ok == "not_found":
        raise HTTPException(404, "Responsibility not found")
    if ok == "already_ended":
        raise HTTPException(400, "Already completed — use Manual Time to edit")
    if ok == "not_started":
        raise HTTPException(400, "Start the timer first, or enter start time manually")
    if ok == "missing_start":
        raise HTTPException(400, "Start time is required when entering an end time")
    if ok == "invalid_range":
        raise HTTPException(400, "End time cannot be earlier than start time")
    if ok == "invalid_time":
        raise HTTPException(400, "Invalid time. Use YYYY-MM-DD HH:MM")
    if ok == "already_active":
        raise HTTPException(
            409,
            "You already have an Active timer on another responsibility — pause or end it first",
        )
    if ok == "paused":
        raise HTTPException(400, "Timer is paused — use Resume")
    raise HTTPException(400, str(ok))


@router.post("/tasks/{responsibility_id}/start")
def post_start_responsibility_timer(responsibility_id: int, body: ResponsibilityTimerIn, request: Request):
    scope = _scope_from_request(request)
    assert_responsibility_in_scope(scope, responsibility_id)
    _, name = _recorder_from_request(request)
    return _timer_http_result(
        start_responsibility_timer(
            responsibility_id,
            body.log_date,
            allow_override=scope.can_edit_assignments,
            actor=name,
        )
    )


@router.post("/tasks/{responsibility_id}/pause")
def post_pause_responsibility_timer(responsibility_id: int, body: ResponsibilityTimerIn, request: Request):
    scope = _scope_from_request(request)
    assert_responsibility_in_scope(scope, responsibility_id)
    _, name = _recorder_from_request(request)
    return _timer_http_result(
        pause_responsibility_timer(
            responsibility_id,
            body.log_date,
            allow_override=scope.can_edit_assignments,
            actor=name,
        )
    )


@router.post("/tasks/{responsibility_id}/resume")
def post_resume_responsibility_timer(responsibility_id: int, body: ResponsibilityTimerIn, request: Request):
    scope = _scope_from_request(request)
    assert_responsibility_in_scope(scope, responsibility_id)
    _, name = _recorder_from_request(request)
    return _timer_http_result(
        resume_responsibility_timer(
            responsibility_id,
            body.log_date,
            allow_override=scope.can_edit_assignments,
            actor=name,
        )
    )


@router.post("/tasks/{responsibility_id}/end")
def post_end_responsibility_timer(responsibility_id: int, body: ResponsibilityTimerIn, request: Request):
    scope = _scope_from_request(request)
    assert_responsibility_in_scope(scope, responsibility_id)
    _, name = _recorder_from_request(request)
    return _timer_http_result(
        end_responsibility_timer(
            responsibility_id,
            body.log_date,
            allow_override=scope.can_edit_assignments,
            actor=name,
        )
    )


@router.get("/tasks/{responsibility_id}/timer")
def get_timer_detail(responsibility_id: int, log_date: str, request: Request):
    scope = _scope_from_request(request)
    assert_responsibility_in_scope(scope, responsibility_id)
    detail = get_responsibility_timer_detail(responsibility_id, log_date)
    if not detail:
        return {
            "timer_status": "Not Started",
            "started_at": "",
            "ended_at": "",
            "paused_at": "",
            "active_seconds": 0,
            "paused_seconds": 0,
            "timer_events": [],
            "daily_time": [],
        }
    return detail


@router.post("/tasks/{responsibility_id}/manual-time")
def post_responsibility_manual_time(responsibility_id: int, body: ResponsibilityTimerIn, request: Request):
    scope = _scope_from_request(request)
    assert_responsibility_in_scope(scope, responsibility_id)
    # Employees may only adjust their own window times; HOD/Admin override allowed
    _, name = _recorder_from_request(request)
    return _timer_http_result(
        set_responsibility_manual_time(
            responsibility_id,
            body.log_date,
            body.started_at,
            body.ended_at,
            allow_override=scope.can_edit_assignments,
            actor=name,
        )
    )


@router.post("/tasks/logs/{log_id}/approve")
def post_approve_task_log(log_id: int, body: TaskApproveIn, request: Request):
    """Linked person (or HOD/Admin) approves or cancels a Done/Partial mark."""
    scope = _scope_from_request(request)
    _, name = _recorder_from_request(request)
    ok = approve_task_log(
        log_id,
        actor=body.approved_by or name,
        linked_employee_id=scope.employee_id,
        action=body.action or "Approved",
        notes=body.notes or "",
        allow_override=scope.can_edit_assignments,
    )
    if ok is True:
        return {"ok": True}
    if ok == "self_forbidden":
        raise HTTPException(403, "You cannot approve or cancel your own responsibility")
    if ok == "forbidden":
        raise HTTPException(403, "Only the Linked To person (or HOD/Admin) can approve")
    if ok == "not_pending":
        raise HTTPException(409, "Task is not awaiting linked approval")
    if ok == "window_closed":
        raise HTTPException(409, "Approval window closed")
    if ok == "invalid_action":
        raise HTTPException(400, "action must be Approved or Cancelled")
    raise HTTPException(404, "Task log not found")


@router.post("/tasks/reassign-day")
def post_reassign_day(body: ReassignDayIn, request: Request):
    """One-day mandatory reassignment clone (HOD/Admin). Master responsibility stays put."""
    scope = _scope_from_request(request)
    assert_hrm_hod_or_admin(scope)
    assert_responsibility_in_scope(scope, body.original_responsibility_id)
    assert_employee_in_scope(scope, body.to_employee_id)
    _, name = _recorder_from_request(request)
    try:
        cid = reassign_mandatory_for_day(
            original_responsibility_id=body.original_responsibility_id,
            to_employee_id=body.to_employee_id,
            reassignment_date=body.reassignment_date,
            assigned_by=body.assigned_by or name,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "clone_id": cid}


@router.post("/tasks/reassign-clones/{clone_id}/mark")
def post_mark_reassignment(clone_id: int, body: ReassignMarkIn, request: Request):
    scope = _scope_from_request(request)
    _, name = _recorder_from_request(request)
    ok = mark_reassignment_clone(
        clone_id,
        body.status,
        marked_by=body.marked_by or name,
        remarks=body.remarks or "",
    )
    if ok is True:
        return {"ok": True}
    if ok == "window_closed":
        raise HTTPException(409, "Reassignment mark window closed")
    if ok == "invalid_status":
        raise HTTPException(400, "Invalid status")
    raise HTTPException(404, "Reassignment not found")


@router.post("/tasks/process-auto-closures")
def post_process_auto_closures(request: Request):
    """HOD/Admin: auto-Missed + auto-Approve linked after IST task date + 2 days."""
    scope = _scope_from_request(request)
    assert_hrm_hod_or_admin(scope)
    _, name = _recorder_from_request(request)
    return process_auto_closures_ist(actor=name or "system-auto")


@router.get("/tasks/logs")
def get_logs(
    request: Request,
    department_id: Optional[int] = None,
    employee_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    scope = _scope_from_request(request)
    dept_f, emp_f = hrm_scope_filters(scope, department_id=department_id, employee_id=employee_id)
    if emp_f == -1 or dept_f == -1:
        return []
    return get_task_logs(dept_f, emp_f, from_date, to_date)


@router.get("/issues/users")
def get_issue_users(request: Request, q: Optional[str] = None, limit: int = 100):
    """Active ERP users for Employee / Caused By searchable pickers."""
    _scope_from_request(request)  # must be authenticated
    return search_active_users(q or "", limit=min(max(limit, 1), 500))


@router.get("/issues/meta")
def get_issue_meta(request: Request):
    _scope_from_request(request)
    return {
        "statuses": ["Open", "Resolve", "Hold", "Cancel"],
        "status_colors": {
            "Open": "blue",
            "Resolve": "green",
            "Hold": "orange",
            "Cancel": "red",
        },
        "issue_types": [
            "General",
            "Discipline",
            "Quality",
            "Attendance",
            "Behaviour",
            "Task Failure",
            "Dependency Missed",
            "Policy Violation",
            "Workplace Incident",
            "Performance",
            "Complaint",
        ],
        "severities": ["Minor", "Moderate", "Major"],
    }


@router.get("/issues")
def get_issues(
    request: Request,
    employee_id: Optional[int] = None,
    department_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    status: Optional[str] = None,
    caused_by_employee_id: Optional[int] = None,
    caused_by_user_id: Optional[int] = None,
    recorded_by_user_id: Optional[int] = None,
    recorded_by: Optional[str] = None,
    subject_user_id: Optional[int] = None,
    designation: Optional[str] = None,
    q: Optional[str] = None,
):
    scope = _scope_from_request(request)
    dept_f, emp_f = hrm_scope_filters(scope, department_id=department_id, employee_id=employee_id)
    if emp_f == -1 or dept_f == -1:
        return []
    _enforce_list_employee_scope(scope, emp_f)
    return list_issues(
        emp_f,
        dept_f,
        from_date,
        to_date,
        status=status,
        caused_by_employee_id=caused_by_employee_id,
        caused_by_user_id=caused_by_user_id,
        recorded_by_user_id=recorded_by_user_id,
        recorded_by=recorded_by,
        subject_user_id=subject_user_id,
        designation=designation,
        q=q,
    )


@router.get("/issues/notifications")
def get_issue_notifications(request: Request, limit: int = 50, unread_only: bool = False):
    _scope_from_request(request)
    return list_issue_notifications(limit=limit, unread_only=unread_only)


@router.get("/issues/{issue_id}/history")
def get_issue_history(issue_id: int, request: Request):
    scope = _scope_from_request(request)
    from ..db.hrm_db import get_issue_employee_id

    eid = get_issue_employee_id(issue_id)
    if eid is None:
        raise HTTPException(404, "Issue not found")
    assert_employee_in_scope(scope, eid)
    return list_issue_history(issue_id)


@router.get("/issues/{issue_id}/comments")
def get_issue_comments_route(issue_id: int, request: Request):
    scope = _scope_from_request(request)
    from ..db.hrm_db import get_issue_employee_id

    eid = get_issue_employee_id(issue_id)
    if eid is None:
        raise HTTPException(404, "Issue not found")
    assert_employee_in_scope(scope, eid)
    return list_issue_comments(issue_id)


@router.post("/issues/{issue_id}/comments")
def post_issue_comment(issue_id: int, body: IssueCommentIn, request: Request):
    scope = _scope_from_request(request)
    from ..db.hrm_db import get_issue_employee_id

    eid = get_issue_employee_id(issue_id)
    if eid is None:
        raise HTTPException(404, "Issue not found")
    assert_employee_in_scope(scope, eid)
    uid, uname = _recorder_from_request(request)
    try:
        cid = add_issue_comment(issue_id, body.comment_text, user_id=uid, user_name=uname)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "id": cid}


@router.get("/issues/{issue_id}/attachments")
def get_issue_attachments_route(issue_id: int, request: Request):
    scope = _scope_from_request(request)
    from ..db.hrm_db import get_issue_employee_id

    eid = get_issue_employee_id(issue_id)
    if eid is None:
        raise HTTPException(404, "Issue not found")
    assert_employee_in_scope(scope, eid)
    return list_issue_attachments(issue_id)


@router.post("/issues/{issue_id}/attachments")
def post_issue_attachment(issue_id: int, body: IssueAttachmentIn, request: Request):
    scope = _scope_from_request(request)
    if not scope.can_edit_issues:
        raise HTTPException(403, "Not permitted to add attachments")
    from ..db.hrm_db import get_issue_employee_id

    eid = get_issue_employee_id(issue_id)
    if eid is None:
        raise HTTPException(404, "Issue not found")
    assert_employee_in_scope(scope, eid)
    uid, uname = _recorder_from_request(request)
    try:
        aid = add_issue_attachment(
            issue_id, body.model_dump(), user_id=uid, user_name=uname
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "id": aid}


@router.post("/issues")
def post_issue(body: IssueIn, request: Request):
    scope = _scope_from_request(request)
    if not scope.can_create_issues:
        raise HTTPException(403, "Not permitted to create issues")
    payload = body.model_dump()
    # Never trust client recorded_by
    payload.pop("recorded_by", None)
    uid, uname = _recorder_from_request(request)
    payload["recorded_by"] = uname
    payload["recorded_by_user_id"] = uid

    employee_id = body.employee_id
    subject_user_id = body.subject_user_id
    if subject_user_id:
        try:
            resolved = _resolve_subject_from_user(subject_user_id, employee_id)
        except HTTPException:
            raise
        payload["employee_id"] = resolved["employee_id"]
        payload["subject_user_id"] = resolved["subject_user_id"]
        payload["subject_user_name"] = resolved["subject_user_name"]
        if not payload.get("department_id") and resolved.get("department_id"):
            payload["department_id"] = resolved["department_id"]
        if not payload.get("designation") and resolved.get("designation"):
            payload["designation"] = resolved["designation"]
        employee_id = resolved["employee_id"]
    if not employee_id:
        raise HTTPException(400, "employee_id or subject_user_id is required")
    assert_employee_in_scope(scope, int(employee_id))

    if body.caused_by_user_id:
        cu = get_user_by_id(int(body.caused_by_user_id))
        if not cu or not int(cu.get("active") or 0):
            raise HTTPException(400, "Caused By user is inactive or not found")
        payload["caused_by_user_id"] = int(cu["id"])
        payload["caused_by_user_name"] = (cu.get("full_name") or cu.get("username") or "").strip()
        if cu.get("employee_id"):
            payload["caused_by_employee_id"] = int(cu["employee_id"])
            try:
                assert_employee_in_scope(scope, int(cu["employee_id"]))
            except HTTPException:
                # Caused-by may be outside scope (cross-dept) when recording for own team — allow name store
                pass
    elif body.caused_by_employee_id:
        assert_employee_in_scope(scope, body.caused_by_employee_id)

    try:
        iid = create_issue(payload)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "id": iid, "recorded_by": uname, "recorded_by_user_id": uid}


@router.patch("/issues/{issue_id}")
def patch_issue(issue_id: int, body: IssueUpdateIn, request: Request):
    scope = _scope_from_request(request)
    if not scope.can_edit_issues:
        raise HTTPException(403, "Not permitted to edit issues")
    from ..db.hrm_db import get_issue_employee_id

    eid = get_issue_employee_id(issue_id)
    if eid is None:
        raise HTTPException(404, "Issue not found")
    assert_employee_in_scope(scope, eid)
    uid, uname = _recorder_from_request(request)
    data = body.model_dump(exclude_unset=True)
    # Strip immutable fields if client tries
    data.pop("recorded_by", None)
    data.pop("recorded_by_user_id", None)
    if data.get("subject_user_id"):
        resolved = _resolve_subject_from_user(int(data["subject_user_id"]), data.get("employee_id"))
        data["employee_id"] = resolved["employee_id"]
        data["subject_user_id"] = resolved["subject_user_id"]
        data["subject_user_name"] = resolved["subject_user_name"]
        assert_employee_in_scope(scope, int(data["employee_id"]))
    elif data.get("employee_id"):
        assert_employee_in_scope(scope, int(data["employee_id"]))
    if data.get("caused_by_user_id"):
        cu = get_user_by_id(int(data["caused_by_user_id"]))
        if not cu or not int(cu.get("active") or 0):
            raise HTTPException(400, "Caused By user is inactive or not found")
        data["caused_by_user_id"] = int(cu["id"])
        data["caused_by_user_name"] = (cu.get("full_name") or cu.get("username") or "").strip()
        if cu.get("employee_id"):
            data["caused_by_employee_id"] = int(cu["employee_id"])
    try:
        result = update_issue(issue_id, data, user_id=uid, user_name=uname)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return result


@router.patch("/issues/{issue_id}/status")
def patch_issue_status(issue_id: int, body: IssueStatusIn, request: Request):
    scope = _scope_from_request(request)
    if not scope.can_change_issue_status:
        raise HTTPException(403, "Not permitted to change issue status")
    from ..db.hrm_db import get_issue_employee_id

    eid = get_issue_employee_id(issue_id)
    if eid is None:
        raise HTTPException(404, "Issue not found")
    assert_employee_in_scope(scope, eid)
    uid, uname = _recorder_from_request(request)
    try:
        return update_issue_status(
            issue_id,
            body.status,
            resolution=body.resolution or "",
            user_id=uid,
            user_name=uname,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.patch("/issues/{issue_id}/resolve")
def patch_resolve_issue(issue_id: int, body: IssueResolveIn, request: Request):
    scope = _scope_from_request(request)
    assert_hrm_hod_or_admin(scope)
    from ..db.hrm_db import get_issue_employee_id

    eid = get_issue_employee_id(issue_id)
    if eid is None:
        raise HTTPException(404, "Issue not found")
    assert_employee_in_scope(scope, eid)
    uid, uname = _recorder_from_request(request)
    resolve_issue(issue_id, body.resolution or "", user_id=uid, user_name=uname)
    return {"ok": True, "status": "Resolve"}


@router.delete("/issues/{issue_id}")
def delete_issue_route(issue_id: int, request: Request):
    scope = _scope_from_request(request)
    if not scope.can_delete_issues:
        raise HTTPException(403, "Only Admin can cancel/delete issues via this action")
    from ..db.hrm_db import get_issue_employee_id

    eid = get_issue_employee_id(issue_id)
    if eid is None:
        raise HTTPException(404, "Issue not found")
    assert_employee_in_scope(scope, eid)
    uid, uname = _recorder_from_request(request)
    delete_issue(issue_id, user_id=uid, user_name=uname)
    return {"ok": True, "status": "Cancel"}


@router.post("/issues/voice-log")
def post_voice_log(body: VoiceTranscriptionIn, request: Request):
    """Audit voice transcription (success or failure). Failures never block issue create."""
    uid, uname = _recorder_from_request(request)
    tid = log_voice_transcription(
        {
            **body.model_dump(),
            "user_id": uid,
            "user_name": uname,
        }
    )
    return {"ok": True, "id": tid}


@router.get("/hod-dashboard/{department_id}")
def hod_dashboard(
    department_id: int,
    request: Request,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    employee_id: Optional[int] = None,
):
    scope = _scope_from_request(request)
    assert_department_in_scope(scope, department_id)
    if employee_id is not None:
        assert_employee_in_scope(scope, employee_id)
    return get_hod_dashboard(department_id, from_date, to_date, employee_id=employee_id)


@router.get("/appraisal/{employee_id}")
def appraisal(
    employee_id: int,
    request: Request,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    scope = _scope_from_request(request)
    assert_employee_in_scope(scope, employee_id)
    data = get_appraisal(employee_id, from_date, to_date)
    if not data:
        raise HTTPException(404, "Employee not found")
    return data


@router.get("/employee-check/{employee_id}")
def employee_day_check(
    employee_id: int,
    request: Request,
    check_date: Optional[str] = None,
):
    """What the employee worked on vs did not for a given day (default today)."""
    scope = _scope_from_request(request)
    assert_employee_in_scope(scope, employee_id)
    data = get_employee_day_check(employee_id, check_date)
    if not data:
        raise HTTPException(404, "Employee not found")
    return data


@router.get("/dwr")
def get_dwr(
    request: Request,
    employee_id: Optional[int] = None,
    department_id: Optional[int] = None,
    check_date: Optional[str] = None,
):
    """Admin/HOD Daily Work Report for a selected employee and date."""
    scope = _scope_from_request(request)
    assert_hrm_hod_or_admin(scope)
    if employee_id is not None:
        assert_employee_in_scope(scope, employee_id)
    if department_id is not None:
        assert_department_in_scope(scope, department_id)
    if department_id is None and employee_id is None and scope.department_id:
        department_id = scope.department_id
    return list_dwr_rows(
        employee_id=employee_id,
        department_id=department_id,
        check_date=check_date,
    )


@router.post("/employee-check/{employee_id}/mark-unmarked-missed")
def employee_mark_unmarked_missed(
    employee_id: int,
    request: Request,
    check_date: Optional[str] = None,
):
    """HOD/Admin: auto-mark unmarked Daily responsibilities as Missed for the day."""
    scope = _scope_from_request(request)
    assert_hrm_hod_or_admin(scope)
    assert_employee_in_scope(scope, employee_id)
    user = getattr(request.state, "user", None) or {}
    marked_by = str(user.get("full_name") or user.get("sub") or user.get("username") or "admin")
    return mark_unmarked_daily_as_missed(employee_id, check_date, marked_by=marked_by)


@router.get("/performance")
def performance(
    request: Request,
    department_id: Optional[int] = None,
    employee_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    scope = _scope_from_request(request)
    dept_f, emp_f = hrm_scope_filters(scope, department_id=department_id, employee_id=employee_id)
    if dept_f == -1 or emp_f == -1:
        return []
    _enforce_list_employee_scope(scope, emp_f)
    rows = get_performance(dept_f, from_date, to_date)
    if emp_f is not None and emp_f > 0:
        rows = [r for r in rows if int(r.get("employee_id") or 0) == int(emp_f)]
    return rows


@router.get("/one-time-tasks")
def get_one_time_tasks(
    request: Request,
    employee_id: Optional[int] = None,
    department_id: Optional[int] = None,
    status: Optional[str] = None,
):
    scope = _scope_from_request(request)
    dept_f, emp_f = hrm_scope_filters(scope, department_id=department_id, employee_id=employee_id)
    if emp_f == -1 or dept_f == -1:
        return []
    _enforce_list_employee_scope(scope, emp_f)
    return list_one_time_tasks(emp_f, dept_f, status=status)


@router.post("/one-time-tasks")
def post_one_time_task(body: OneTimeTaskIn, request: Request):
    scope = _scope_from_request(request)
    assert_employee_in_scope(scope, body.employee_id)
    assert_employee_in_scope(scope, body.backup_employee_id)
    if scope.is_employee:
        raise HTTPException(403, "Employees cannot assign one-time tasks")
    if int(body.backup_employee_id) == int(body.employee_id):
        raise HTTPException(400, "Backup person must be different from the assigned employee")
    if float(body.backup_allocation_value or 0) <= 0:
        raise HTTPException(400, "Backup allocation duration must be greater than zero")
    data = body.model_dump()
    data["require_backup"] = True
    if not (data.get("assigned_by") or "").strip():
        _, name = _recorder_from_request(request)
        data["assigned_by"] = name
    try:
        tid = create_one_time_task(data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "id": tid}


@router.patch("/one-time-tasks/{task_id}")
def patch_one_time_task(task_id: int, body: OneTimeTaskUpdate, request: Request):
    scope = _scope_from_request(request)
    assert_hrm_admin_mutate_records(scope)
    owner = get_one_time_task_owner(task_id)
    if owner is None:
        raise HTTPException(404, "Task not found")
    assert_employee_in_scope(scope, owner)
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    if "employee_id" in data:
        assert_employee_in_scope(scope, int(data["employee_id"]))
    update_one_time_task(task_id, data)
    return {"ok": True}


@router.delete("/one-time-tasks/{task_id}")
def del_one_time_task(task_id: int, request: Request):
    scope = _scope_from_request(request)
    owner = get_one_time_task_owner(task_id)
    if owner is None:
        raise HTTPException(404, "Task not found")
    assert_employee_in_scope(scope, owner)
    assert_hrm_delete_allowed(scope)
    cancel_one_time_task(task_id)
    return {"ok": True}


@router.post("/one-time-tasks/{task_id}/start")
def post_start_one_time_task(task_id: int, request: Request):
    scope = _scope_from_request(request)
    owner = get_one_time_task_owner(task_id)
    if owner is None:
        raise HTTPException(404, "Task not found")
    assert_employee_in_scope(scope, owner)
    if scope.is_employee and scope.employee_id != owner:
        raise HTTPException(403, "You can only start your own tasks")
    if not start_one_time_task(task_id):
        raise HTTPException(400, "Task cannot be started (must be Pending or Rejected)")
    return {"ok": True}


@router.post("/one-time-tasks/{task_id}/complete")
def post_complete_one_time_task(task_id: int, body: OneTimeTaskNotesIn, request: Request):
    scope = _scope_from_request(request)
    owner = get_one_time_task_owner(task_id)
    if owner is None:
        raise HTTPException(404, "Task not found")
    assert_employee_in_scope(scope, owner)
    if scope.is_employee and scope.employee_id != owner:
        raise HTTPException(403, "You can only complete your own tasks")
    if not complete_one_time_task(task_id, body.notes or ""):
        raise HTTPException(400, "Task must be In Progress to mark complete")
    return {"ok": True}


@router.post("/one-time-tasks/{task_id}/approve")
def post_approve_one_time_task(task_id: int, body: OneTimeTaskApprovalIn, request: Request):
    scope = _scope_from_request(request)
    owner = get_one_time_task_owner(task_id)
    if owner is None:
        raise HTTPException(404, "Task not found")
    assert_employee_in_scope(scope, owner)
    if scope.is_employee:
        raise HTTPException(403, "HOD approval required")
    if not approve_one_time_task(task_id, body.approved_by or "", body.notes or ""):
        raise HTTPException(400, "Task must be Done to approve")
    return {"ok": True}


@router.post("/one-time-tasks/{task_id}/reject")
def post_reject_one_time_task(task_id: int, body: OneTimeTaskApprovalIn, request: Request):
    scope = _scope_from_request(request)
    owner = get_one_time_task_owner(task_id)
    if owner is None:
        raise HTTPException(404, "Task not found")
    assert_employee_in_scope(scope, owner)
    if scope.is_employee:
        raise HTTPException(403, "HOD approval required")
    if not reject_one_time_task(task_id, body.approved_by or "", body.notes or ""):
        raise HTTPException(400, "Task must be Done to reject")
    return {"ok": True}


@router.post("/one-time-tasks/{task_id}/manual-duration")
def post_manual_duration(task_id: int, body: ManualDurationIn, request: Request):
    scope = _scope_from_request(request)
    owner = get_one_time_task_owner(task_id)
    if owner is None:
        raise HTTPException(404, "Task not found")
    assert_employee_in_scope(scope, owner)
    # Assignee or assigners with edit rights can set duration
    if scope.is_employee and scope.employee_id != owner and not scope.can_edit_assignments:
        raise HTTPException(403, "Not allowed to set duration")
    try:
        mins = set_manual_task_duration(task_id, body.duration)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "duration_minutes": mins}


@router.get("/dashboard-stats")
def dashboard_stats(request: Request):
    scope = _scope_from_request(request)
    dept_f, emp_f = hrm_scope_filters(scope)
    if dept_f == -1 or emp_f == -1:
        return {"department_count": 0, "total_employees": 0, "hods": []}
    return get_dashboard_stats(
        department_id=dept_f if scope.level == "department" else None,
        employee_id=emp_f if scope.level == "self" else None,
    )


@router.get("/hierarchy")
def hierarchy_get(request: Request):
    scope = _scope_from_request(request)
    if not scope.can_manage_org and not scope.is_hod:
        raise HTTPException(403, "Hierarchy view requires HOD or Admin")
    tree = get_org_hierarchy()
    if scope.level == "department" and scope.department_id is not None:
        did = scope.department_id
        tree["employees_flat"] = [e for e in tree["employees_flat"] if e.get("department_id") == did]
        tree["departments_flat"] = [d for d in tree["departments_flat"] if d.get("id") == did]
        tree["departments"] = [d for d in tree["departments"] if d.get("id") == did]
        tree["reporting"] = [
            e for e in tree["reporting"] if e.get("department_id") == did
        ]
    return tree


@router.patch("/hierarchy/department")
def hierarchy_patch_dept(body: HierarchyDeptIn, request: Request):
    assert_hrm_write_org(_scope_from_request(request))
    data = {}
    if body.parent_department_id is not None:
        data["parent_department_id"] = body.parent_department_id
    if body.hod_name is not None:
        data["hod_name"] = body.hod_name
    if data:
        update_department(body.department_id, data)
    return {"ok": True}


@router.patch("/hierarchy/reporting")
def hierarchy_patch_report(body: HierarchyReportIn, request: Request):
    assert_hrm_write_org(_scope_from_request(request))
    try:
        update_employee(
            body.employee_id,
            {"reports_to_employee_id": body.reports_to_employee_id},
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@router.get("/reports/tasks")
def task_based_report(
    request: Request,
    department_id: Optional[int] = None,
    employee_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
):
    scope = _scope_from_request(request)
    dept_f, emp_f = hrm_scope_filters(scope, department_id=department_id, employee_id=employee_id)
    if dept_f == -1 or emp_f == -1:
        return []
    _enforce_list_employee_scope(scope, emp_f)
    return get_task_based_report(
        department_id=dept_f,
        employee_id=emp_f,
        from_date=from_date,
        to_date=to_date,
        priority=priority,
        status=status,
    )


@router.get("/assignees")
def list_assignees(request: Request, q: str = ""):
    """Authorized users for Assigned By dropdown."""
    _scope_from_request(request)
    users = search_active_users(q or "", limit=50)
    return [
        {
            "id": u.get("id"),
            "name": (u.get("full_name") or u.get("username") or "").strip(),
            "username": u.get("username"),
            "role": u.get("role_name"),
        }
        for u in users
        if (u.get("full_name") or u.get("username"))
    ]
