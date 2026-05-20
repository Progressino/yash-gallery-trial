"""HRM Module Router — task tracking, issues, appraisal."""
from fastapi import APIRouter, HTTPException
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
)

router = APIRouter()


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


@router.get("/departments")
def get_departments():
    return list_departments()


@router.post("/departments")
def post_department(body: DepartmentIn):
    create_department(body.model_dump())
    return {"ok": True}


@router.patch("/departments/{did}")
def patch_department(did: int, body: DepartmentUpdate):
    update_department(did, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}


@router.get("/employees")
def get_employees(department_id: Optional[int] = None, status: str = "Active"):
    return list_employees(department_id, status)


@router.post("/employees")
def post_employee(body: EmployeeIn):
    code = create_employee(body.model_dump())
    return {"ok": True, "emp_code": code}


@router.patch("/employees/{eid}")
def patch_employee(eid: int, body: EmployeeUpdate):
    update_employee(eid, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}


@router.get("/responsibilities")
def get_responsibilities(employee_id: Optional[int] = None, department_id: Optional[int] = None):
    return list_responsibilities(employee_id, department_id)


@router.post("/responsibilities")
def post_responsibility(body: ResponsibilityIn):
    create_responsibility(body.model_dump())
    return {"ok": True}


@router.patch("/responsibilities/{rid}")
def patch_responsibility(rid: int, body: ResponsibilityUpdate):
    update_responsibility(rid, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}


@router.delete("/responsibilities/{rid}")
def del_responsibility(rid: int):
    delete_responsibility(rid)
    return {"ok": True}


@router.post("/tasks/mark")
def post_mark_task(body: TaskMarkIn):
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
    department_id: Optional[int] = None,
    employee_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    return get_task_logs(department_id, employee_id, from_date, to_date)


@router.get("/issues")
def get_issues(
    employee_id: Optional[int] = None,
    department_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    return list_issues(employee_id, department_id, from_date, to_date)


@router.post("/issues")
def post_issue(body: IssueIn):
    create_issue(body.model_dump())
    return {"ok": True}


@router.patch("/issues/{issue_id}/resolve")
def patch_resolve_issue(issue_id: int, body: IssueResolveIn):
    resolve_issue(issue_id, body.resolution)
    return {"ok": True}


@router.get("/hod-dashboard/{department_id}")
def hod_dashboard(department_id: int, from_date: Optional[str] = None, to_date: Optional[str] = None):
    return get_hod_dashboard(department_id, from_date, to_date)


@router.get("/appraisal/{employee_id}")
def appraisal(employee_id: int, from_date: Optional[str] = None, to_date: Optional[str] = None):
    data = get_appraisal(employee_id, from_date, to_date)
    if not data:
        raise HTTPException(404, "Employee not found")
    return data


@router.get("/performance")
def performance(
    department_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    return get_performance(department_id, from_date, to_date)
