"""HRM Module Router — task tracking, issues, appraisal (RBAC-scoped)."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from ..db.hrm_db import (
    list_departments,
    create_department,
    update_department,
    list_employees,
    create_employee,
    update_employee,
    list_responsibilities,
    create_responsibility,
    update_responsibility,
    delete_responsibility,
    mark_task,
    get_task_logs,
    list_issues,
    create_issue,
    resolve_issue,
    get_hod_dashboard,
    get_appraisal,
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
)
from ..db.users_db import get_user_auth_profile
from ..services.rbac import (
    build_hrm_scope,
    hrm_scope_filters,
    assert_department_in_scope,
    assert_employee_in_scope,
    assert_hrm_write_org,
    assert_hrm_hod_or_admin,
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


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    department_id: Optional[int] = None
    designation: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    join_date: Optional[str] = None
    status: Optional[str] = None


class ResponsibilityIn(BaseModel):
    employee_id: int
    department_id: Optional[int] = None
    title: str
    description: Optional[str] = ""
    frequency: Optional[str] = "Daily"
    category: Optional[str] = "General"
    added_by: Optional[str] = ""


class ResponsibilityUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    frequency: Optional[str] = None
    category: Optional[str] = None
    employee_id: Optional[int] = None
    active: Optional[int] = None


class TaskMarkIn(BaseModel):
    responsibility_id: int
    log_date: str
    status: str = "Done"
    marked_by: Optional[str] = ""
    remarks: Optional[str] = ""
    blocker_employee_id: Optional[int] = None
    blocker_reason: Optional[str] = ""


class IssueIn(BaseModel):
    employee_id: int
    department_id: Optional[int] = None
    issue_date: Optional[str] = None
    issue_type: Optional[str] = "General"
    severity: Optional[str] = "Minor"
    title: str
    description: Optional[str] = ""
    recorded_by: Optional[str] = ""
    caused_by_employee_id: Optional[int] = None
    caused_by_dept_id: Optional[int] = None


class IssueResolveIn(BaseModel):
    resolution: str


class OneTimeTaskIn(BaseModel):
    employee_id: int
    department_id: Optional[int] = None
    title: str
    description: Optional[str] = ""
    due_date: Optional[str] = ""
    assigned_by: Optional[str] = ""


class OneTimeTaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[str] = None
    employee_id: Optional[int] = None


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
    code = create_employee(body.model_dump())
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
    update_employee(eid, data)
    return {"ok": True}


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
    return list_responsibilities(emp_f, dept_f)


@router.post("/responsibilities")
def post_responsibility(body: ResponsibilityIn, request: Request):
    scope = _scope_from_request(request)
    assert_employee_in_scope(scope, body.employee_id)
    if scope.is_employee and scope.employee_id != body.employee_id:
        raise HTTPException(403, "Cannot assign responsibilities for other employees")
    create_responsibility(body.model_dump())
    return {"ok": True}


@router.patch("/responsibilities/{rid}")
def patch_responsibility(rid: int, body: ResponsibilityUpdate, request: Request):
    scope = _scope_from_request(request)
    assert_hrm_hod_or_admin(scope)
    from ..db.hrm_db import get_responsibility_owner

    owner = get_responsibility_owner(rid)
    if owner is None:
        raise HTTPException(404, "Responsibility not found")
    assert_employee_in_scope(scope, owner)
    if body.employee_id is not None:
        assert_employee_in_scope(scope, body.employee_id)
    update_responsibility(rid, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}


@router.delete("/responsibilities/{rid}")
def del_responsibility(rid: int, request: Request):
    scope = _scope_from_request(request)
    assert_hrm_hod_or_admin(scope)
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
    )
    if not ok:
        raise HTTPException(404, "Responsibility not found")
    return {"ok": True}


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


@router.get("/issues")
def get_issues(
    request: Request,
    employee_id: Optional[int] = None,
    department_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    scope = _scope_from_request(request)
    dept_f, emp_f = hrm_scope_filters(scope, department_id=department_id, employee_id=employee_id)
    if emp_f == -1 or dept_f == -1:
        return []
    return list_issues(emp_f, dept_f, from_date, to_date)


@router.post("/issues")
def post_issue(body: IssueIn, request: Request):
    scope = _scope_from_request(request)
    assert_employee_in_scope(scope, body.employee_id)
    if body.caused_by_employee_id:
        assert_employee_in_scope(scope, body.caused_by_employee_id)
    create_issue(body.model_dump())
    return {"ok": True}


@router.patch("/issues/{issue_id}/resolve")
def patch_resolve_issue(issue_id: int, body: IssueResolveIn, request: Request):
    scope = _scope_from_request(request)
    from ..db.hrm_db import get_issue_employee_id

    eid = get_issue_employee_id(issue_id)
    if eid is None:
        raise HTTPException(404, "Issue not found")
    assert_employee_in_scope(scope, eid)
    resolve_issue(issue_id, body.resolution)
    return {"ok": True}


@router.get("/hod-dashboard/{department_id}")
def hod_dashboard(
    department_id: int,
    request: Request,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    scope = _scope_from_request(request)
    assert_department_in_scope(scope, department_id)
    return get_hod_dashboard(department_id, from_date, to_date)


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


@router.get("/performance")
def performance(
    request: Request,
    department_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    scope = _scope_from_request(request)
    dept_f, emp_f = hrm_scope_filters(scope, department_id=department_id)
    if dept_f == -1:
        return []
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
    return list_one_time_tasks(emp_f, dept_f, status=status)


@router.post("/one-time-tasks")
def post_one_time_task(body: OneTimeTaskIn, request: Request):
    scope = _scope_from_request(request)
    assert_employee_in_scope(scope, body.employee_id)
    if scope.is_employee:
        raise HTTPException(403, "Employees cannot assign one-time tasks")
    tid = create_one_time_task(body.model_dump())
    return {"ok": True, "id": tid}


@router.patch("/one-time-tasks/{task_id}")
def patch_one_time_task(task_id: int, body: OneTimeTaskUpdate, request: Request):
    scope = _scope_from_request(request)
    assert_hrm_hod_or_admin(scope)
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
    if scope.is_employee:
        raise HTTPException(403, "Employees cannot cancel assigned tasks")
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
