import { useState, useEffect, useMemo, useRef, type ReactNode } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import { useAuth } from '../store/auth'
import { FREQUENCIES, PRIORITIES, TIME_PERIODS, WEEKDAYS, MONTHS, priorityStyle } from './hrmConstants'
import { loadHrmLang, saveHrmLang, t, type HrmLang } from './hrmI18n'

type Tab = 'dashboard' | 'check' | 'employees' | 'responsibilities' | 'tasks' | 'hod' | 'issues' | 'appraisal' | 'performance' | 'hierarchy'

const ONE_TIME_STATUSES = ['Pending', 'In Progress', 'Done', 'Approved', 'Rejected'] as const
const TASK_LOG_STATUSES = ['Done', 'Partial', 'Missed', 'Blocked', 'Leave', 'N/A'] as const
const CATEGORIES = ['General', 'Quality', 'Production', 'Accounts', 'Purchase', 'Sales', 'Store', 'HR', 'Other']
const ISSUE_TYPES = ['General', 'Discipline', 'Quality', 'Attendance', 'Behaviour', 'Task Failure', 'Dependency Missed', 'Policy Violation', 'Workplace Incident', 'Performance', 'Complaint']
const SEVERITIES = ['Minor', 'Moderate', 'Major']
const ISSUE_STATUSES = ['Open', 'Resolve', 'Hold', 'Cancel'] as const

/** Column header with inline filter control (reserved for dense tables) */
function _ColFilter({ label, children }: { label: string; children: ReactNode }) {
  return (
    <th className="text-left px-2 py-1.5 align-bottom">
      <div className="text-[10px] font-semibold text-gray-500 uppercase">{label}</div>
      <div className="mt-0.5">{children}</div>
    </th>
  )
}
void _ColFilter

const issueStatusStyle = (s: string) => {
  if (s === 'Resolve' || s === 'Resolved') return 'bg-green-100 text-green-800'
  if (s === 'Open') return 'bg-blue-100 text-blue-800'
  if (s === 'Hold') return 'bg-orange-100 text-orange-800'
  if (s === 'Cancel') return 'bg-red-100 text-red-800'
  return 'bg-gray-100 text-gray-600'
}

const today = () => new Date().toISOString().split('T')[0]
const fmt7Days = () => { const d = new Date(); d.setDate(d.getDate() - 6); return d.toISOString().split('T')[0] }
const fmtMonth = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-01` }

const statusLabel = (s: string) => {
  if (s === 'Done') return '✅'
  if (s === 'Partial') return '⚠️'
  if (s === 'Missed') return '❌'
  if (s === 'Blocked') return '🔴'
  if (s === 'Leave') return '🏖'
  if (s === 'N/A') return '—'
  return '○'
}

const statusBg = (s: string) => {
  if (s === 'Done') return 'bg-green-500 text-white'
  if (s === 'Partial') return 'bg-yellow-400 text-white'
  if (s === 'Missed') return 'bg-red-500 text-white'
  if (s === 'Blocked') return 'bg-purple-600 text-white'
  if (s === 'Leave') return 'bg-slate-500 text-white'
  if (s === 'N/A') return 'bg-gray-500 text-white'
  return 'bg-gray-200 text-gray-500'
}

const oneTimeStatusStyle = (s: string) => {
  if (s === 'Pending') return 'bg-gray-100 text-gray-700'
  if (s === 'In Progress') return 'bg-blue-100 text-blue-700'
  if (s === 'Done') return 'bg-amber-100 text-amber-800'
  if (s === 'Approved') return 'bg-green-100 text-green-700'
  if (s === 'Rejected') return 'bg-red-100 text-red-700'
  return 'bg-gray-100 text-gray-600'
}

const fmtDuration = (mins: number) => {
  if (!mins || mins <= 0) return '—'
  if (mins < 60) return `${mins}m`
  const h = Math.floor(mins / 60)
  const m = mins % 60
  return m ? `${h}h ${m}m` : `${h}h`
}

const fmtDate = (iso: string) => {
  if (!iso) return '—'
  const d = new Date(iso.replace(' ', 'T'))
  if (Number.isNaN(d.getTime())) return iso.split(' ')[0] || iso
  return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })
}

const fmtDateTime = (iso: string) => {
  if (!iso) return '—'
  const d = new Date(iso.replace(' ', 'T'))
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
}

function toDatetimeLocal(s?: string) {
  if (!s) return ''
  return String(s).replace(' ', 'T').slice(0, 16)
}

function timerBadgeClass(st: string) {
  if (st === 'Completed') return 'bg-green-100 text-green-800'
  if (st === 'In Progress') return 'bg-amber-100 text-amber-800'
  return 'bg-gray-100 text-gray-600'
}

function LinkedPersonLine({ item }: { item: any }) {
  const name = item?.linked_to_employee_name
  return (
    <p className="text-xs text-indigo-800 mt-0.5">
      Supervisor / Approver: {name ? <b>{name}</b> : <span className="text-gray-500">Self-complete</span>}
    </p>
  )
}

export default function HRM() {
  const qc = useQueryClient()
  const authUser = useAuth(s => s.user)
  const { data: scopeApi } = useQuery({
    queryKey: ['hrm-scope'],
    queryFn: () => api.get('/hrm/scope').then(r => r.data),
  })
  const scope = scopeApi || authUser?.hrm_scope
  const userRole = scope?.role || authUser?.role || ''
  const canManageOrg = scope?.can_manage_org ?? (userRole === 'Super Admin' || userRole === 'Admin' || userRole === 'Sir')
  const canViewEmployeeList = scope?.can_view_employee_list ?? (userRole === 'Super Admin' || userRole === 'Admin')
  const scopeLevel = scope?.level || 'all'
  const isEmployeeScope = scopeLevel === 'self'
  const canAssignTasks = !isEmployeeScope
  const canEditAssignments = scope?.can_edit_assignments ?? (canManageOrg || userRole === 'HOD')
  const canMutateRecords = scope?.can_mutate_assignment_records ?? canManageOrg
  const canDeleteHrm = scope?.can_delete_hrm_records ?? canMutateRecords
  const canUseEmployeeCheck = scope?.can_use_employee_check ?? (userRole === 'HOD' || userRole === 'Employee' || !canViewEmployeeList)
  const canViewDashboard = scope?.can_view_dashboard ?? (canManageOrg || userRole === 'HOD')

  const [lang, setLang] = useState<HrmLang>(() => loadHrmLang())
  const setLangPersist = (l: HrmLang) => { setLang(l); saveHrmLang(l) }

  const [tab, setTab] = useState<Tab>('dashboard')
  const [selDept, setSelDept] = useState<number | ''>('')
  const [selEmp, setSelEmp] = useState<number | ''>('')
  const [hodDept, setHodDept] = useState<number | ''>('')
  const [hodEmp, setHodEmp] = useState<number | ''>('')
  const [appraisalEmp, setAppraisalEmp] = useState<number | ''>('')
  const [fromDate, setFromDate] = useState(fmt7Days())
  const [toDate, setToDate] = useState(today())
  const [appraisalFrom, setAppraisalFrom] = useState(fmtMonth())
  const [appraisalTo, setAppraisalTo] = useState(today())
  const [checkDate, setCheckDate] = useState(today())
  const [checkEmp, setCheckEmp] = useState<number | ''>('')
  const [checkPeriod, setCheckPeriod] = useState('')
  const [showDailyGuide, setShowDailyGuide] = useState(false)

  const [showDeptForm, setShowDeptForm] = useState(false)
  const [showEmpForm, setShowEmpForm] = useState(false)
  const [showRespForm, setShowRespForm] = useState(false)
  const [showTaskForm, setShowTaskForm] = useState(false)
  const [showIssueForm, setShowIssueForm] = useState(false)
  const [taskStatusFilter, setTaskStatusFilter] = useState('')
  const [taskPriorityFilter, setTaskPriorityFilter] = useState('')
  const [taskTitleFilter, setTaskTitleFilter] = useState('')
  const [taskAssignedByFilter, setTaskAssignedByFilter] = useState('')
  const [respTitleFilter, setRespTitleFilter] = useState('')
  const [respFreqFilter, setRespFreqFilter] = useState('')
  const [respPriorityFilter, setRespPriorityFilter] = useState('')
  const [respAssignedByFilter, setRespAssignedByFilter] = useState('')
  const [manualTimeOpen, setManualTimeOpen] = useState<number | null>(null)
  const [manualTimeVal, setManualTimeVal] = useState('')
  const [dwrManualId, setDwrManualId] = useState<number | null>(null)
  const [dwrManualStart, setDwrManualStart] = useState('')
  const [dwrManualEnd, setDwrManualEnd] = useState('')
  const [empNameSuggest, setEmpNameSuggest] = useState<any[]>([])
  const [audioPreview, setAudioPreview] = useState<string | null>(null)
  void audioPreview
  const issueAudioRecRef = useRef<MediaRecorder | null>(null)
  const issueAudioChunks = useRef<Blob[]>([])

  const [completeModal, setCompleteModal] = useState<{ id: number; title: string } | null>(null)
  const [completeNotes, setCompleteNotes] = useState('')
  const [approvalModal, setApprovalModal] = useState<{ id: number; title: string; action: 'approve' | 'reject' } | null>(null)
  const [approvalNotes, setApprovalNotes] = useState('')
  const [hodSubTab, setHodSubTab] = useState<'responsibilities' | 'tasks' | 'dwr'>('responsibilities')
  const [editDept, setEditDept] = useState<any>(null)
  const [editEmp, setEditEmp] = useState<any>(null)
  const [editResp, setEditResp] = useState<any>(null)
  const [editTask, setEditTask] = useState<any>(null)

  // Blocked modal
  const [blockedModal, setBlockedModal] = useState<{ respId: number; date: string } | null>(null)
  const [blockedForm, setBlockedForm] = useState({ blocker_employee_id: '' as any, blocker_reason: '', marked_by: '' })

  const [deptForm, setDeptForm] = useState({ name: '', description: '', hod_name: '' })
  const [empForm, setEmpForm] = useState({ name: '', emp_code: '', department_id: '' as any, designation: '', phone: '', email: '', join_date: '', reports_to_employee_id: '' as any })
  const [respForm, setRespForm] = useState({
    employee_id: '' as any, title: '', description: '', frequency: 'Daily', category: 'General',
    added_by: '', priority: 'Medium', mandatory: false, schedule_weekday: '', schedule_month_day: 0,
    schedule_month: 0, time_period: '', linked_to_employee_id: '' as any,
  })
  const [taskForm, setTaskForm] = useState({
    employee_id: '' as any, title: '', description: '', due_date: '', assigned_by: '', priority: 'Medium',
  })
  const [reassignForm, setReassignForm] = useState({
    original_responsibility_id: '' as any, to_employee_id: '' as any, reassignment_date: today(),
  })
  const [showReassign, setShowReassign] = useState(false)
  const [issueForm, setIssueForm] = useState({
    subject_user_id: '' as any,
    employee_id: '' as any,
    issue_type: 'General',
    severity: 'Minor',
    title: '',
    description: '',
    caused_by_user_id: '' as any,
    caused_by_employee_id: '' as any,
    status: 'Open',
    audio_url: '',
  })
  const [issueEmpSearch, setIssueEmpSearch] = useState('')
  const [issueCauseSearch, setIssueCauseSearch] = useState('')
  const [issueStatusFilter, setIssueStatusFilter] = useState('')
  const [issueQ, setIssueQ] = useState('')
  const [editIssue, setEditIssue] = useState<any | null>(null)
  const [issueHistory, setIssueHistory] = useState<any[]>([])
  const [showHistoryId, setShowHistoryId] = useState<number | null>(null)
  const [issueVoiceTarget, setIssueVoiceTarget] = useState<'title' | 'description'>('description')
  const [issueListening, setIssueListening] = useState(false)
  const [issueVoiceStatus, setIssueVoiceStatus] = useState('')
  const issueRecRef = useRef<any>(null)
  const [issueComment, setIssueComment] = useState('')
  const [issueAttachmentName, setIssueAttachmentName] = useState('')

  // Quick assign (dashboard) — separate responsibility vs task flows without Item Type label
  const [quickResp, setQuickResp] = useState({
    mode: 'responsibility' as 'responsibility' | 'task',
    employee_id: '' as any, department_id: '' as any, title: '', description: '',
    frequency: 'Daily', category: 'General', added_by: '', due_date: '',
    priority: 'Medium', mandatory: false, schedule_weekday: '', schedule_month_day: 0,
    schedule_month: 0, time_period: '', linked_to_employee_id: '' as any,
  })
  const [showQuickResp, setShowQuickResp] = useState(false)
  const [voiceText, setVoiceText] = useState('')
  const [isListening, setIsListening] = useState(false)
  const [aiParsing, setAiParsing] = useState(false)
  const [aiParsed, setAiParsed] = useState<any>(null)

  const respImportRef = useRef<HTMLInputElement>(null)
  const taskImportRef = useRef<HTMLInputElement>(null)

  const parseLocally = (text: string) => {
    setAiParsing(true)
    const lowerText = text.toLowerCase()
    let matchedEmp: any = null
    let bestScore = 0
    ;(allEmps as any[]).forEach((e: any) => {
      const nameParts = e.name.toLowerCase().split(' ')
      let score = 0
      nameParts.forEach((part: string) => { if (part.length > 2 && lowerText.includes(part)) score++ })
      if (score > bestScore) { bestScore = score; matchedEmp = e }
    })
    let frequency = 'Daily'
    if (lowerText.includes('weekly')) frequency = 'Weekly'
    else if (lowerText.includes('fortnight')) frequency = 'Fortnightly'
    else if (lowerText.includes('monthly')) frequency = 'Monthly'
    else if (lowerText.includes('quarter')) frequency = 'Quarterly'
    else if (lowerText.includes('yearly') || lowerText.includes('annual')) frequency = 'Yearly'
    const taskHints = ['by friday', 'by monday', 'by tuesday', 'by wednesday', 'by thursday', 'by saturday', 'by sunday', 'one time', 'one-time', 'audit', 'complete the', 'finish the', 'before ']
    let mode: 'responsibility' | 'task' = taskHints.some(h => lowerText.includes(h)) ? 'task' : 'responsibility'
    let dueDate = ''
    if (mode === 'task') {
      const dayMatch = lowerText.match(/by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)/i)
      if (dayMatch) {
        const days = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
        const target = days.indexOf(dayMatch[1].toLowerCase())
        const d = new Date()
        const diff = (target - d.getDay() + 7) % 7 || 7
        d.setDate(d.getDate() + diff)
        dueDate = d.toISOString().split('T')[0]
      }
    }
    let title = text
    if (matchedEmp) {
      matchedEmp.name.split(' ').forEach((part: string) => {
        title = title.replace(new RegExp(part, 'gi'), '').trim()
      })
    }
    title = title.replace(/\b(from now on|daily|weekly|monthly|fortnightly|quarterly|yearly|will|shall|must|the|and|or)\b/gi, '').replace(/\s+/g, ' ').trim()
    if (!title) title = text
    const parsed = { mode, employee_id: matchedEmp?.id || null, employee_name: matchedEmp?.name || '', department_id: matchedEmp?.department_id || null, title, frequency, category: 'General', due_date: dueDate }
    setAiParsed(parsed)
    setQuickResp(f => ({
      ...f,
      mode,
      employee_id: parsed.employee_id || '',
      department_id: parsed.department_id || '',
      title: parsed.title,
      description: '',
      frequency: parsed.frequency,
      category: 'General',
      due_date: dueDate,
    }))
    setShowQuickResp(true)
    setAiParsing(false)
  }

  const startListening = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) { alert('Speech recognition requires Chrome or Edge.'); return }
    const recognition = new SR()
    recognition.lang = lang === 'hi' ? 'hi-IN' : 'en-IN'
    recognition.continuous = false
    recognition.interimResults = false
    recognition.onstart = () => setIsListening(true)
    recognition.onend = () => setIsListening(false)
    recognition.onresult = (e: any) => { const text = e.results[0][0].transcript; setVoiceText(text); parseLocally(text) }
    recognition.onerror = () => setIsListening(false)
    recognition.start()
  }

  useEffect(() => {
    if (!scope) return
    if (scope.level === 'self' && scope.employee_id) {
      setSelEmp(scope.employee_id)
      setAppraisalEmp(scope.employee_id)
      setCheckEmp(scope.employee_id)
      if (scope.department_id) setSelDept(scope.department_id)
    }
    if (scope.level === 'department' && scope.department_id) {
      setSelDept(scope.department_id)
      setHodDept(scope.department_id)
    }
  }, [scope?.level, scope?.employee_id, scope?.department_id])

  // ── Queries ──────────────────────────────────────────────────────────────────
  const { data: depts = [] } = useQuery({ queryKey: ['hrm-depts'], queryFn: () => api.get('/hrm/departments').then(r => r.data) })
  const { data: employees = [] } = useQuery({ queryKey: ['hrm-emps', selDept], queryFn: () => api.get(`/hrm/employees${selDept ? `?department_id=${selDept}` : ''}`).then(r => r.data) })
  const { data: allEmps = [] } = useQuery({ queryKey: ['hrm-all-emps'], queryFn: () => api.get('/hrm/employees').then(r => r.data) })
  const { data: responsibilities = [] } = useQuery({
    queryKey: ['hrm-resps', selDept, selEmp],
    queryFn: () => api.get(`/hrm/responsibilities${selDept ? `?department_id=${selDept}` : ''}${selEmp ? `${selDept ? '&' : '?'}employee_id=${selEmp}` : ''}`).then(r => r.data)
  })
  const { data: hodData } = useQuery({
    queryKey: ['hrm-hod', hodDept, hodEmp, fromDate, toDate],
    queryFn: () => api.get(`/hrm/hod-dashboard/${hodDept}?from_date=${fromDate}&to_date=${toDate}${hodEmp ? `&employee_id=${hodEmp}` : ''}`).then(r => r.data),
    enabled: !!hodDept,
  })
  const { data: dwrData } = useQuery({
    queryKey: ['hrm-dwr', hodDept, hodEmp, toDate],
    queryFn: () => {
      const p = new URLSearchParams({ check_date: toDate })
      if (hodEmp) p.set('employee_id', String(hodEmp))
      if (hodDept) p.set('department_id', String(hodDept))
      return api.get(`/hrm/dwr?${p}`).then(r => r.data)
    },
    enabled: tab === 'hod' && hodSubTab === 'dwr' && !!hodDept,
  })
  const { data: issues = [] } = useQuery({
    queryKey: ['hrm-issues', selDept, selEmp, fromDate, toDate, issueStatusFilter, issueQ],
    queryFn: () => {
      const p = new URLSearchParams()
      if (selDept) p.set('department_id', String(selDept))
      if (selEmp) p.set('employee_id', String(selEmp))
      if (fromDate) p.set('from_date', fromDate)
      if (toDate) p.set('to_date', toDate)
      if (issueStatusFilter) p.set('status', issueStatusFilter)
      if (issueQ.trim()) p.set('q', issueQ.trim())
      return api.get(`/hrm/issues?${p.toString()}`).then(r => r.data)
    },
    enabled: tab === 'issues',
  })
  const { data: issueUsers = [] } = useQuery({
    queryKey: ['hrm-issue-users', issueEmpSearch || issueCauseSearch],
    queryFn: () => api.get('/hrm/issues/users', {
      params: { q: issueEmpSearch || issueCauseSearch || '', limit: 80 },
    }).then(r => r.data),
    enabled: tab === 'issues' && (showIssueForm || !!editIssue),
  })
  const { data: issueUsersForFilter = [] } = useQuery({
    queryKey: ['hrm-issue-users-all'],
    queryFn: () => api.get('/hrm/issues/users', { params: { limit: 200 } }).then(r => r.data),
    enabled: tab === 'issues',
  })
  const { data: appraisalData } = useQuery({
    queryKey: ['hrm-appraisal', appraisalEmp, appraisalFrom, appraisalTo],
    queryFn: () => api.get(`/hrm/appraisal/${appraisalEmp}?from_date=${appraisalFrom}&to_date=${appraisalTo}`).then(r => r.data),
    enabled: !!appraisalEmp,
  })
  const { data: dayCheck, isFetching: dayCheckLoading } = useQuery({
    queryKey: ['hrm-employee-check', checkEmp, checkDate],
    queryFn: () => api.get(`/hrm/employee-check/${checkEmp}?check_date=${checkDate}`).then(r => r.data),
    enabled: !!checkEmp && (tab === 'check' || tab === 'hod' || (tab === 'dashboard' && isEmployeeScope)),
    refetchInterval: tab === 'check' ? 60_000 : false,
  })
  const markMissedMut = useMutation({
    mutationFn: () => api.post(`/hrm/employee-check/${checkEmp}/mark-unmarked-missed?check_date=${checkDate}`).then(r => r.data),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ['hrm-employee-check'] })
      qc.invalidateQueries({ queryKey: ['hrm-hod'] })
      qc.invalidateQueries({ queryKey: ['hrm-appraisal'] })
      qc.invalidateQueries({ queryKey: ['hrm-performance'] })
      window.alert(`Marked ${data?.marked ?? 0} unmarked daily item(s) as Missed.`)
    },
  })
  const { data: perfData = [] } = useQuery({
    queryKey: ['hrm-perf', selDept, selEmp, fromDate, toDate],
    queryFn: () => api.get(`/hrm/performance?from_date=${fromDate}&to_date=${toDate}${selDept ? `&department_id=${selDept}` : ''}${selEmp ? `&employee_id=${selEmp}` : ''}`).then(r => r.data),
    enabled: tab === 'performance',
  })
  const myTaskEmpId = isEmployeeScope ? scope?.employee_id : null
  const { data: myTasks = [] } = useQuery({
    queryKey: ['hrm-my-tasks', myTaskEmpId],
    queryFn: () => api.get(`/hrm/one-time-tasks?employee_id=${myTaskEmpId}`).then(r => r.data),
    enabled: tab === 'dashboard' && !!myTaskEmpId,
  })
  const { data: oneTimeTasks = [] } = useQuery({
    queryKey: ['hrm-one-time-tasks', selDept, selEmp, taskStatusFilter],
    queryFn: () => {
      const params = new URLSearchParams()
      if (selDept) params.set('department_id', String(selDept))
      if (selEmp) params.set('employee_id', String(selEmp))
      if (taskStatusFilter) params.set('status', taskStatusFilter)
      const q = params.toString()
      return api.get(`/hrm/one-time-tasks${q ? `?${q}` : ''}`).then(r => r.data)
    },
    enabled: tab === 'tasks' && !isEmployeeScope,
  })
  const { data: hodPendingTasks = [] } = useQuery({
    queryKey: ['hrm-hod-pending-tasks', hodDept],
    queryFn: () => api.get(`/hrm/one-time-tasks?department_id=${hodDept}&status=Done`).then(r => r.data),
    enabled: tab === 'hod' && !!hodDept,
  })

  // ── Mutations ─────────────────────────────────────────────────────────────────
  const createDeptMut = useMutation({ mutationFn: (b: object) => api.post('/hrm/departments', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-depts'] }); setShowDeptForm(false); setDeptForm({ name: '', description: '', hod_name: '' }) } })
  const updateDeptMut = useMutation({ mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/hrm/departments/${id}`, data), onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-depts'] }); setEditDept(null) } })
  const createEmpMut = useMutation({ mutationFn: (b: object) => api.post('/hrm/employees', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-emps'] }); qc.invalidateQueries({ queryKey: ['hrm-all-emps'] }); setShowEmpForm(false) } })
  const updateEmpMut = useMutation({ mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/hrm/employees/${id}`, data), onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-emps'] }); qc.invalidateQueries({ queryKey: ['hrm-all-emps'] }); setEditEmp(null) } })
  const deleteEmpMut = useMutation({ mutationFn: (id: number) => api.delete(`/hrm/employees/${id}`), onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-emps'] }); qc.invalidateQueries({ queryKey: ['hrm-all-emps'] }) } })
  const createRespMut = useMutation({ mutationFn: (b: object) => api.post('/hrm/responsibilities', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-resps'] }); qc.invalidateQueries({ queryKey: ['hrm-hod'] }); setShowRespForm(false); setShowQuickResp(false); setAiParsed(null); setVoiceText('') } })
  const updateRespMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/hrm/responsibilities/${id}`, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-resps'] }); qc.invalidateQueries({ queryKey: ['hrm-hod'] }); setEditResp(null) },
  })
  const deleteRespMut = useMutation({ mutationFn: (id: number) => api.delete(`/hrm/responsibilities/${id}`), onSuccess: () => qc.invalidateQueries({ queryKey: ['hrm-resps'] }) })
  const markTaskMut = useMutation({
    mutationFn: (b: object) => api.post('/hrm/tasks/mark', b),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['hrm-hod'] })
      qc.invalidateQueries({ queryKey: ['hrm-perf'] })
      qc.invalidateQueries({ queryKey: ['hrm-employee-check'] })
      qc.invalidateQueries({ queryKey: ['hrm-dwr'] })
      setBlockedModal(null)
    },
    onError: (err: any) => {
      const msg = err?.response?.data?.detail
      if (msg) alert(msg)
    },
  })
  const reassignMut = useMutation({
    mutationFn: (b: object) => api.post('/hrm/tasks/reassign-day', b),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['hrm-employee-check'] })
      setShowReassign(false)
    },
    onError: (err: any) => alert(err?.response?.data?.detail || 'Reassignment failed'),
  })
  const markCloneMut = useMutation({
    mutationFn: ({ id, ...b }: { id: number; status: string; remarks?: string }) =>
      api.post(`/hrm/tasks/reassign-clones/${id}/mark`, b),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hrm-employee-check'] }),
    onError: (err: any) => alert(err?.response?.data?.detail || 'Could not mark reassignment'),
  })
  const approveLogMut = useMutation({
    mutationFn: ({ id, action }: { id: number; action: string }) =>
      api.post(`/hrm/tasks/logs/${id}/approve`, { action }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['hrm-employee-check'] })
      qc.invalidateQueries({ queryKey: ['hrm-appraisal'] })
      qc.invalidateQueries({ queryKey: ['hrm-perf'] })
      qc.invalidateQueries({ queryKey: ['hrm-dwr'] })
    },
    onError: (err: any) => alert(err?.response?.data?.detail || 'Approval failed'),
  })
  const invalidateDwr = () => {
    qc.invalidateQueries({ queryKey: ['hrm-employee-check'] })
    qc.invalidateQueries({ queryKey: ['hrm-hod'] })
    qc.invalidateQueries({ queryKey: ['hrm-dwr'] })
  }
  const startRespMut = useMutation({
    mutationFn: ({ id, log_date }: { id: number; log_date: string }) =>
      api.post(`/hrm/tasks/${id}/start`, { log_date }),
    onSuccess: invalidateDwr,
    onError: (err: any) => alert(err?.response?.data?.detail || 'Could not start timer'),
  })
  const endRespMut = useMutation({
    mutationFn: ({ id, log_date }: { id: number; log_date: string }) =>
      api.post(`/hrm/tasks/${id}/end`, { log_date }),
    onSuccess: invalidateDwr,
    onError: (err: any) => alert(err?.response?.data?.detail || 'Could not end timer'),
  })
  const manualRespTimeMut = useMutation({
    mutationFn: ({ id, log_date, started_at, ended_at }: { id: number; log_date: string; started_at: string; ended_at: string }) =>
      api.post(`/hrm/tasks/${id}/manual-time`, { log_date, started_at, ended_at }),
    onSuccess: () => {
      invalidateDwr()
      setDwrManualId(null)
    },
    onError: (err: any) => alert(err?.response?.data?.detail || 'Could not save time'),
  })
  const importRespMut = useMutation({
    mutationFn: (file: File) => {
      const fd = new FormData()
      fd.append('file', file)
      return api.post('/hrm/import/responsibilities', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
    },
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['hrm-resps'] })
      qc.invalidateQueries({ queryKey: ['hrm-hod'] })
      const { created, errors } = res.data
      alert(`Imported ${created} responsibility row(s).${errors?.length ? `\n\nIssues:\n${errors.slice(0, 5).join('\n')}` : ''}`)
    },
  })
  const importTaskMut = useMutation({
    mutationFn: (file: File) => {
      const fd = new FormData()
      fd.append('file', file)
      return api.post('/hrm/import/one-time-tasks', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
    },
    onSuccess: (res) => {
      invalidateTaskMetrics()
      const { created, errors } = res.data
      alert(`Imported ${created} task row(s).${errors?.length ? `\n\nIssues:\n${errors.slice(0, 5).join('\n')}` : ''}`)
    },
  })
  const invalidateIssues = () => {
    qc.invalidateQueries({ queryKey: ['hrm-issues'] })
    qc.invalidateQueries({ queryKey: ['hrm-appraisal'] })
    qc.invalidateQueries({ queryKey: ['hrm-perf'] })
  }
  const createIssueMut = useMutation({
    mutationFn: (b: object) => api.post('/hrm/issues', b),
    onSuccess: () => {
      invalidateIssues()
      setShowIssueForm(false)
      setIssueForm({
        subject_user_id: '', employee_id: '', issue_type: 'General', severity: 'Minor',
        title: '', description: '', caused_by_user_id: '', caused_by_employee_id: '', status: 'Open',
        audio_url: '',
      })
    },
    onError: (e: any) => alert(e?.response?.data?.detail || 'Failed to create issue'),
  })
  const resolveIssueMut = useMutation({
    mutationFn: ({ id, res }: { id: number; res: string }) => api.patch(`/hrm/issues/${id}/resolve`, { resolution: res }),
    onSuccess: () => invalidateIssues(),
  })
  const updateIssueMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/hrm/issues/${id}`, data),
    onSuccess: () => { invalidateIssues(); setEditIssue(null) },
    onError: (e: any) => alert(e?.response?.data?.detail || 'Update failed'),
  })
  const statusIssueMut = useMutation({
    mutationFn: ({ id, status, resolution }: { id: number; status: string; resolution?: string }) =>
      api.patch(`/hrm/issues/${id}/status`, { status, resolution: resolution || '' }),
    onSuccess: () => invalidateIssues(),
    onError: (e: any) => alert(e?.response?.data?.detail || 'Status change failed'),
  })
  const commentIssueMut = useMutation({
    mutationFn: ({ id, text }: { id: number; text: string }) =>
      api.post(`/hrm/issues/${id}/comments`, { comment_text: text }),
    onSuccess: () => { setIssueComment(''); invalidateIssues() },
  })
  const attachIssueMut = useMutation({
    mutationFn: ({ id, file_name }: { id: number; file_name: string }) =>
      api.post(`/hrm/issues/${id}/attachments`, { file_name }),
    onSuccess: () => { setIssueAttachmentName(''); invalidateIssues() },
  })

  const stopIssueVoice = () => {
    try { issueRecRef.current?.stop?.() } catch { /* ignore */ }
    setIssueListening(false)
  }

  const startIssueVoice = (target: 'title' | 'description', mode: 'append' | 'replace' = 'append') => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) {
      setIssueVoiceStatus('Speech recognition requires Chrome or Edge. Type manually.')
      api.post('/hrm/issues/voice-log', {
        transcript: '', target_field: target, status: 'failed',
        error_message: 'SpeechRecognition unsupported',
      }).catch(() => {})
      return
    }
    if (issueListening && issueRecRef.current) {
      stopIssueVoice()
      setIssueVoiceStatus('Paused')
      return
    }
    setIssueVoiceTarget(target)
    const recognition = new SR()
    recognition.lang = 'en-IN'
    recognition.continuous = true
    recognition.interimResults = true
    let finalChunk = ''
    recognition.onstart = () => {
      setIssueListening(true)
      setIssueVoiceStatus(`Listening for ${target}…`)
    }
    recognition.onend = () => {
      setIssueListening(false)
      if (finalChunk.trim()) {
        const apply = (prev: string) => {
          if (mode === 'replace' || !prev.trim()) return finalChunk.trim()
          return `${prev.trim()} ${finalChunk.trim()}`.trim()
        }
        setIssueForm(f => target === 'title'
          ? { ...f, title: apply(f.title) }
          : { ...f, description: apply(f.description) })
        if (editIssue) {
          setEditIssue((f: any) => target === 'title'
            ? { ...f, title: apply(f.title || '') }
            : { ...f, description: apply(f.description || '') })
        }
        api.post('/hrm/issues/voice-log', {
          transcript: finalChunk.trim(),
          target_field: target,
          status: 'success',
          issue_id: editIssue?.id || null,
        }).catch(() => {})
        setIssueVoiceStatus('Transcribed — edit freely')
      } else {
        setIssueVoiceStatus('Stopped')
      }
    }
    recognition.onerror = (ev: any) => {
      setIssueListening(false)
      setIssueVoiceStatus(`Voice error: ${ev?.error || 'failed'} — type manually`)
      api.post('/hrm/issues/voice-log', {
        transcript: finalChunk,
        target_field: target,
        status: 'failed',
        error_message: String(ev?.error || 'error'),
        issue_id: editIssue?.id || null,
      }).catch(() => {})
    }
    recognition.onresult = (e: any) => {
      let interim = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript
        if (e.results[i].isFinal) finalChunk += `${t} `
        else interim += t
      }
      if (interim) setIssueVoiceStatus(`Listening… ${interim}`)
    }
    issueRecRef.current = recognition
    recognition.start()
  }

  const openIssueHistory = async (id: number) => {
    if (showHistoryId === id) {
      setShowHistoryId(null)
      return
    }
    const { data } = await api.get(`/hrm/issues/${id}/history`)
    setIssueHistory(data || [])
    setShowHistoryId(id)
  }

  const recordedByName = authUser?.full_name || authUser?.username || 'You'
  const canEditIssues = canEditAssignments
  const canChangeIssueStatus = canEditAssignments
  const invalidateTaskMetrics = () => {
    qc.invalidateQueries({ queryKey: ['hrm-one-time-tasks'] })
    qc.invalidateQueries({ queryKey: ['hrm-my-tasks'] })
    qc.invalidateQueries({ queryKey: ['hrm-hod-pending-tasks'] })
    qc.invalidateQueries({ queryKey: ['hrm-appraisal'] })
    qc.invalidateQueries({ queryKey: ['hrm-perf'] })
  }
  const updateOneTimeTaskMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/hrm/one-time-tasks/${id}`, data),
    onSuccess: () => { invalidateTaskMetrics(); setEditTask(null) },
  })
  const createOneTimeTaskMut = useMutation({
    mutationFn: (b: object) => api.post('/hrm/one-time-tasks', b),
    onSuccess: () => {
      invalidateTaskMetrics()
      setShowTaskForm(false)
      setShowQuickResp(false)
      setShowRespForm(false)
      setAiParsed(null)
      setVoiceText('')
      setTaskForm({ employee_id: '', title: '', description: '', due_date: '', assigned_by: '', priority: 'Medium' })
    },
  })
  const startOneTimeTaskMut = useMutation({
    mutationFn: (id: number) => api.post(`/hrm/one-time-tasks/${id}/start`),
    onSuccess: invalidateTaskMetrics,
  })
  const completeOneTimeTaskMut = useMutation({
    mutationFn: ({ id, notes }: { id: number; notes: string }) => api.post(`/hrm/one-time-tasks/${id}/complete`, { notes }),
    onSuccess: () => {
      invalidateTaskMetrics()
      setCompleteModal(null)
      setCompleteNotes('')
    },
  })
  const approveOneTimeTaskMut = useMutation({
    mutationFn: ({ id, notes }: { id: number; notes: string }) => api.post(`/hrm/one-time-tasks/${id}/approve`, { notes }),
    onSuccess: () => {
      invalidateTaskMetrics()
      setApprovalModal(null)
      setApprovalNotes('')
    },
  })
  const rejectOneTimeTaskMut = useMutation({
    mutationFn: ({ id, notes }: { id: number; notes: string }) => api.post(`/hrm/one-time-tasks/${id}/reject`, { notes }),
    onSuccess: () => {
      invalidateTaskMetrics()
      setApprovalModal(null)
      setApprovalNotes('')
    },
  })
  const cancelOneTimeTaskMut = useMutation({
    mutationFn: (id: number) => api.delete(`/hrm/one-time-tasks/${id}`),
    onSuccess: invalidateTaskMetrics,
  })
  const manualDurationMut = useMutation({
    mutationFn: ({ id, duration }: { id: number; duration: string }) =>
      api.post(`/hrm/one-time-tasks/${id}/manual-duration`, { duration }),
    onSuccess: () => {
      invalidateTaskMetrics()
      setManualTimeOpen(null)
      setManualTimeVal('')
    },
    onError: (err: any) => alert(err?.response?.data?.detail || 'Invalid time format'),
  })
  const submitAssign = (form: typeof quickResp) => {
    if (!form.employee_id || !form.title) return
    if (form.mode === 'task') {
      createOneTimeTaskMut.mutate({
        employee_id: +form.employee_id,
        title: form.title,
        description: form.description || '',
        due_date: form.due_date || '',
        assigned_by: form.added_by || '',
        priority: form.priority || 'Medium',
      })
    } else {
      if ((form.frequency === 'Weekly' || form.frequency === 'Fortnightly') && !form.schedule_weekday) {
        alert('Select a weekday for Weekly/Fortnightly responsibilities')
        return
      }
      if (form.frequency === 'Monthly' && !(form.schedule_month_day > 0)) {
        alert('Select calendar day for Monthly responsibilities')
        return
      }
      if (form.frequency === 'Quarterly' && !(form.schedule_month > 0)) {
        alert('Select anchor month for Quarterly responsibilities')
        return
      }
      createRespMut.mutate({
        employee_id: +form.employee_id,
        department_id: form.department_id ? +form.department_id : null,
        title: form.title,
        description: form.description || '',
        frequency: form.frequency,
        category: form.category,
        added_by: form.added_by || '',
        priority: form.priority || 'Medium',
        mandatory: !!form.mandatory,
        schedule_weekday: form.schedule_weekday || '',
        schedule_month_day: form.schedule_month_day || 0,
        schedule_month: form.schedule_month || 0,
        time_period: form.time_period || '',
        linked_to_employee_id: form.linked_to_employee_id ? +form.linked_to_employee_id : null,
      })
    }
  }

  const hodDeptEmployees = useMemo(
    () => (allEmps as any[]).filter((e: any) => !hodDept || e.department_id === hodDept),
    [allEmps, hodDept],
  )

  const { data: dashStats } = useQuery({
    queryKey: ['hrm-dash-stats'],
    queryFn: () => api.get('/hrm/dashboard-stats').then(r => r.data),
    enabled: tab === 'dashboard',
  })
  const { data: assignees = [] } = useQuery({
    queryKey: ['hrm-assignees'],
    queryFn: () => api.get('/hrm/assignees').then(r => r.data),
    staleTime: 60_000,
  })
  const { data: orgHierarchy } = useQuery({
    queryKey: ['hrm-hierarchy'],
    queryFn: () => api.get('/hrm/hierarchy').then(r => r.data),
    enabled: tab === 'hierarchy' && (canManageOrg || userRole === 'HOD'),
  })
  const { data: taskReport = [] } = useQuery({
    queryKey: ['hrm-task-report', selDept, selEmp, fromDate, toDate, taskPriorityFilter, taskStatusFilter],
    queryFn: () => {
      const p = new URLSearchParams({ from_date: fromDate, to_date: toDate })
      if (selDept) p.set('department_id', String(selDept))
      if (selEmp) p.set('employee_id', String(selEmp))
      if (taskPriorityFilter) p.set('priority', taskPriorityFilter)
      if (taskStatusFilter) p.set('status', taskStatusFilter)
      return api.get(`/hrm/reports/tasks?${p}`).then(r => r.data)
    },
    enabled: tab === 'performance',
  })

  const filteredResponsibilities = useMemo(() => {
    return (responsibilities as any[]).filter(r => {
      if (respTitleFilter && !String(r.title || '').toLowerCase().includes(respTitleFilter.toLowerCase())) return false
      if (respFreqFilter && r.frequency !== respFreqFilter) return false
      if (respPriorityFilter && (r.priority || 'Medium') !== respPriorityFilter) return false
      if (respAssignedByFilter && !String(r.added_by || '').toLowerCase().includes(respAssignedByFilter.toLowerCase())) return false
      return true
    })
  }, [responsibilities, respTitleFilter, respFreqFilter, respPriorityFilter, respAssignedByFilter])

  const filteredTasks = useMemo(() => {
    return (oneTimeTasks as any[]).filter(t => {
      if (taskTitleFilter && !String(t.title || '').toLowerCase().includes(taskTitleFilter.toLowerCase())) return false
      if (taskPriorityFilter && (t.priority || 'Medium') !== taskPriorityFilter) return false
      if (taskAssignedByFilter && !String(t.assigned_by || '').toLowerCase().includes(taskAssignedByFilter.toLowerCase())) return false
      return true
    }).sort((a, b) => {
      const order = { Critical: 0, High: 1, Medium: 2, Low: 3 } as Record<string, number>
      return (order[a.priority || 'Medium'] ?? 2) - (order[b.priority || 'Medium'] ?? 2)
    })
  }, [oneTimeTasks, taskTitleFilter, taskPriorityFilter, taskAssignedByFilter])

  const dayCheckFiltered = useMemo(() => {
    if (!dayCheck || !checkPeriod) return dayCheck
    const filterItems = (arr: any[]) => (arr || []).filter((i: any) => !i.time_period || i.time_period === checkPeriod || i.time_period === 'Full Day')
    return {
      ...dayCheck,
      worked_on: filterItems(dayCheck.worked_on),
      not_worked: filterItems(dayCheck.not_worked),
      other: filterItems(dayCheck.other),
      whenever_required: filterItems(dayCheck.whenever_required || []),
      additional_work: dayCheck.additional_work || [],
    }
  }, [dayCheck, checkPeriod])

  const handleStatusSelect = (respId: number, logDate: string, status: string) => {
    if (!status) return
    if (status === 'Blocked') {
      setBlockedModal({ respId, date: logDate })
      setBlockedForm({ blocker_employee_id: '', blocker_reason: '', marked_by: '' })
      return
    }
    markTaskMut.mutate({ responsibility_id: respId, log_date: logDate, status })
  }

  const renderRespCheck = (i: any, opts?: { showMark?: boolean }) => {
    const showMark = opts?.showMark !== false
    const ts = i.timer_status || 'Not Started'
    const canTime = i.in_action_window !== false || canEditAssignments
    const canMark = showMark && (!i.marked || i.status === 'Pending' || canEditAssignments)
    const openManual = () => {
      setDwrManualId(i.responsibility_id)
      setDwrManualStart(toDatetimeLocal(i.started_at))
      setDwrManualEnd(toDatetimeLocal(i.ended_at))
    }
    return (
      <div className="flex-1 min-w-0">
        <p className="font-medium text-sm text-gray-800">{i.title}</p>
        <p className="text-xs text-gray-400">
          {i.frequency} · {i.status}{i.approval_status ? ` · ${i.approval_status}` : ''}{i.marked_by ? ` · ${i.marked_by}` : ''}{i.blocker_reason ? ` — ${i.blocker_reason}` : ''}
        </p>
        <LinkedPersonLine item={i} />
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full ${timerBadgeClass(ts)}`}>{ts}</span>
          <span className="text-[11px] text-gray-500">Start: {fmtDateTime(i.started_at)}</span>
          <span className="text-[11px] text-gray-500">End: {fmtDateTime(i.ended_at)}</span>
          <span className="text-[11px] font-semibold text-[#002B5B]">{fmtDuration(i.duration_minutes)}</span>
        </div>
        {canTime && (
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            {ts === 'Not Started' && (
              <button type="button" className="text-xs px-2 py-0.5 bg-blue-600 text-white rounded" onClick={() => startRespMut.mutate({ id: i.responsibility_id, log_date: checkDate })}>▶ Start</button>
            )}
            {ts === 'In Progress' && (
              <button type="button" className="text-xs px-2 py-0.5 bg-amber-600 text-white rounded" onClick={() => endRespMut.mutate({ id: i.responsibility_id, log_date: checkDate })}>■ End</button>
            )}
            {dwrManualId === i.responsibility_id ? (
              <div className="flex flex-wrap items-center gap-1 w-full">
                <label className="text-[10px] text-gray-500">Start
                  <input type="datetime-local" value={dwrManualStart} onChange={e => setDwrManualStart(e.target.value)} className="ml-1 border rounded px-1 py-0.5 text-xs" />
                </label>
                <label className="text-[10px] text-gray-500">End
                  <input type="datetime-local" value={dwrManualEnd} onChange={e => setDwrManualEnd(e.target.value)} className="ml-1 border rounded px-1 py-0.5 text-xs" />
                </label>
                <button type="button" className="text-xs px-2 py-0.5 bg-green-700 text-white rounded" onClick={() => {
                  if (dwrManualStart && dwrManualEnd && dwrManualEnd < dwrManualStart) {
                    alert('End time cannot be earlier than start time')
                    return
                  }
                  manualRespTimeMut.mutate({
                    id: i.responsibility_id,
                    log_date: checkDate,
                    started_at: dwrManualStart ? dwrManualStart.replace('T', ' ') : '',
                    ended_at: dwrManualEnd ? dwrManualEnd.replace('T', ' ') : '',
                  })
                }}>Save time</button>
                <button type="button" className="text-xs text-gray-500" onClick={() => setDwrManualId(null)}>Cancel</button>
              </div>
            ) : (
              <button type="button" className="text-[10px] text-blue-600 underline" onClick={openManual}>Manual time</button>
            )}
          </div>
        )}
        {canMark && (
          <select
            defaultValue=""
            onChange={e => {
              const val = e.target.value
              if (!val) return
              handleStatusSelect(i.responsibility_id, checkDate, val)
              e.target.value = ''
            }}
            className="mt-1.5 text-xs border rounded px-1.5 py-1 bg-white"
          >
            <option value="">Mark status…</option>
            {TASK_LOG_STATUSES.map(st => <option key={st} value={st}>{st}</option>)}
          </select>
        )}
        {i.task_log_id && i.approval_status === 'Pending' && (canEditAssignments || Number(scope?.employee_id) === Number(i.linked_to_employee_id)) && (
          <div className="flex gap-1 mt-1">
            <button type="button" className="text-xs px-2 py-0.5 bg-green-700 text-white rounded" onClick={() => approveLogMut.mutate({ id: i.task_log_id, action: 'Approved' })}>Approve</button>
            <button type="button" className="text-xs px-2 py-0.5 bg-red-700 text-white rounded" onClick={() => approveLogMut.mutate({ id: i.task_log_id, action: 'Cancelled' })}>Cancel</button>
          </div>
        )}
      </div>
    )
  }

  const deptName = (id: any) => (depts as any[]).find(d => d.id === id)?.name || '—'

  const ALL_TABS: [Tab, string][] = [
    ['dashboard', `📊 ${t(lang, 'dashboard')}`],
    ['check', `🔎 ${t(lang, 'check')}`],
    ['employees', `👥 ${t(lang, 'employees')}`],
    ['responsibilities', `📋 ${t(lang, 'responsibilities')}`],
    ['tasks', `✅ ${t(lang, 'tasks')}`],
    ['hierarchy', `🏛 ${t(lang, 'hierarchy')}`],
    ['hod', `🏢 ${t(lang, 'hod')}`],
    ['issues', `⚠️ ${t(lang, 'issues')}`],
    ['appraisal', `📁 ${t(lang, 'appraisal')}`],
    ['performance', `📈 ${t(lang, 'performance')}`],
  ]

  const TABS = useMemo(() => {
    let tabs = ALL_TABS
    if (!canViewDashboard) {
      tabs = tabs.filter(([k]) => k !== 'dashboard')
    }
    if (!canViewEmployeeList) {
      tabs = tabs.filter(([k]) => k !== 'employees')
    }
    if (!canUseEmployeeCheck) {
      tabs = tabs.filter(([k]) => k !== 'check')
    }
    if (!canManageOrg && userRole !== 'HOD') {
      tabs = tabs.filter(([k]) => k !== 'hierarchy')
    }
    if (scopeLevel === 'self') {
      tabs = tabs.filter(([k]) => ['check', 'responsibilities', 'issues', 'appraisal'].includes(k) || (canViewDashboard && k === 'dashboard'))
      // Employees never see dashboard (canViewDashboard false for Employee)
      tabs = tabs.filter(([k]) => k !== 'dashboard')
    }
    return tabs
  }, [scopeLevel, canViewEmployeeList, canUseEmployeeCheck, canViewDashboard, canManageOrg, userRole, lang])

  useEffect(() => {
    if (!TABS.some(([k]) => k === tab)) setTab(TABS[0]?.[0] || 'dashboard')
  }, [TABS, tab])

  const scopeHint =
    scopeLevel === 'self'
      ? 'You see only your own responsibilities, tasks, issues, and appraisal.'
      : scopeLevel === 'department'
        ? 'You see your department team only (not the org-wide employees list).'
        : null

  const pickerEmps = useMemo(() => {
    const rows = allEmps as any[]
    if (isEmployeeScope && scope?.employee_id) {
      return rows.filter((e: any) => e.id === scope.employee_id)
    }
    return rows
  }, [allEmps, isEmployeeScope, scope?.employee_id])

  const assigneeOptions = useMemo(() => {
    const names = new Set<string>()
    const opts: string[] = []
    for (const a of assignees as any[]) {
      const n = (a.name || a.username || '').trim()
      if (n && !names.has(n)) { names.add(n); opts.push(n) }
    }
    if (authUser?.full_name && !names.has(authUser.full_name)) opts.unshift(authUser.full_name)
    if (authUser?.username && !names.has(authUser.username)) opts.push(authUser.username)
    return opts
  }, [assignees, authUser])

  const startIssueAudio = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const rec = new MediaRecorder(stream)
      issueAudioChunks.current = []
      rec.ondataavailable = (e) => { if (e.data.size) issueAudioChunks.current.push(e.data) }
      rec.onstop = () => {
        const blob = new Blob(issueAudioChunks.current, { type: 'audio/webm' })
        const url = URL.createObjectURL(blob)
        setAudioPreview(url)
        setIssueForm(f => ({ ...f, audio_url: url }))
        stream.getTracks().forEach(tr => tr.stop())
      }
      issueAudioRecRef.current = rec
      rec.start()
      setIssueVoiceStatus('Recording audio…')
    } catch {
      setIssueVoiceStatus('Microphone access denied')
    }
  }
  const stopIssueAudio = () => {
    issueAudioRecRef.current?.stop()
    setIssueVoiceStatus('Audio ready for attachment / playback')
  }
  void startIssueAudio
  void stopIssueAudio

  const searchEmpNames = async (q: string) => {
    setEmpForm(f => ({ ...f, name: q }))
    if (q.trim().length < 2) { setEmpNameSuggest([]); return }
    try {
      const res = await api.get('/hrm/employees/autocomplete', { params: { q } })
      setEmpNameSuggest(res.data || [])
    } catch { setEmpNameSuggest([]) }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-gray-800">{t(lang, 'title')}</h1>
          <p className="text-sm text-gray-500">
            {t(lang, 'subtitle')}
            {scopeHint && <span className="block text-amber-700 mt-1">{scopeHint}</span>}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">{t(lang, 'lang')}</span>
          <button type="button" onClick={() => setLangPersist('en')}
            className={`text-xs px-2 py-1 rounded border ${lang === 'en' ? 'bg-[#002B5B] text-white' : 'bg-white'}`}>EN</button>
          <button type="button" onClick={() => setLangPersist('hi')}
            className={`text-xs px-2 py-1 rounded border ${lang === 'hi' ? 'bg-[#002B5B] text-white' : 'bg-white'}`}>हिं</button>
        </div>
      </div>

      <div className="flex flex-wrap gap-1 bg-gray-100 p-1 rounded-lg">
        {TABS.map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${tab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* ── DASHBOARD ── */}
      {tab === 'dashboard' && (
        <div className="space-y-4">
          {(dashStats?.department_count ?? 0) > 0 || (dashStats?.total_employees ?? 0) > 0 ? (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <div className="bg-white rounded-xl p-4 border shadow-sm">
                <p className="text-2xl font-bold text-blue-600">{dashStats?.department_count ?? (depts as any[]).length}</p>
                <p className="text-xs text-gray-500 mt-1 font-semibold">{t(lang, 'departments')}</p>
              </div>
              <div className="bg-white rounded-xl p-4 border shadow-sm">
                <p className="text-2xl font-bold text-purple-600">{dashStats?.total_employees ?? (allEmps as any[]).length}</p>
                <p className="text-xs text-gray-500 mt-1 font-semibold">{t(lang, 'totalEmployees')}</p>
              </div>
              <div className="bg-white rounded-xl p-4 border shadow-sm md:col-span-1 col-span-2">
                <p className="text-xs text-gray-500 font-semibold mb-1">{t(lang, 'hodInfo')}</p>
                <ul className="text-sm space-y-0.5 max-h-24 overflow-auto">
                  {((dashStats?.hods) || (depts as any[]).map((d: any) => ({ department: d.name, hod_name: d.hod_name }))).slice(0, 8).map((h: any, i: number) => (
                    <li key={i}><span className="font-medium">{h.department || h.department_name}</span>: {h.hod_name || '—'}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : null}

          {isEmployeeScope && dayCheck && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 bg-teal-700 text-white flex justify-between items-center">
                <h3 className="font-semibold">Today — worked vs not worked</h3>
                <button onClick={() => setTab('check')} className="text-xs bg-white/20 px-2 py-1 rounded">Open full check</button>
              </div>
              <div className="grid md:grid-cols-2 divide-y md:divide-y-0 md:divide-x">
                <div className="p-4">
                  <p className="text-xs font-semibold text-green-700 mb-2">Worked on ({dayCheck.worked_on?.length || 0})</p>
                  {(dayCheck.worked_on || []).length === 0 ? <p className="text-xs text-gray-400">Nothing marked Done/Partial yet.</p> : (
                    <ul className="space-y-1.5">{(dayCheck.worked_on || []).map((i: any) => (
                      <li key={i.responsibility_id} className="text-sm flex gap-2"><span>{statusLabel(i.status)}</span><span>{i.title}</span></li>
                    ))}</ul>
                  )}
                </div>
                <div className="p-4">
                  <p className="text-xs font-semibold text-red-700 mb-2">Not worked / pending ({dayCheck.not_worked?.length || 0})</p>
                  {(dayCheck.not_worked || []).length === 0 ? <p className="text-xs text-gray-400">All clear.</p> : (
                    <ul className="space-y-1.5">{(dayCheck.not_worked || []).map((i: any) => (
                      <li key={i.responsibility_id} className="text-sm flex gap-2"><span>{statusLabel(i.status)}</span><span>{i.title}</span></li>
                    ))}</ul>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* My Tasks — employee dashboard */}
          {isEmployeeScope && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 bg-[#002B5B] text-white flex justify-between items-center">
                <h3 className="font-semibold">✅ My Tasks</h3>
                <span className="text-xs text-blue-200">Start → Done → HOD approval</span>
              </div>
              {(myTasks as any[]).filter((t: any) => t.status !== 'Approved').length === 0 ? (
                <p className="text-center text-gray-400 py-8 text-sm">No active tasks assigned.</p>
              ) : (
                <div className="divide-y">
                  {(myTasks as any[]).filter((t: any) => t.status !== 'Approved').map((t: any) => (
                    <div key={t.id} className="px-4 py-3 flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <p className="font-medium text-gray-800">{t.title}</p>
                        {t.description && <p className="text-xs text-gray-400">{t.description}</p>}
                        <div className="flex flex-wrap gap-2 mt-1 text-xs text-gray-500">
                          <span>Due: {t.due_date || '—'}</span>
                          <span>Time: {fmtDuration(t.duration_minutes)}</span>
                          {t.started_at && <span>Started {fmtDateTime(t.started_at)}</span>}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${oneTimeStatusStyle(t.status)}`}>{t.status}</span>
                        {(t.status === 'Pending' || t.status === 'Rejected') && (
                          <button onClick={() => startOneTimeTaskMut.mutate(t.id)} className="text-xs px-3 py-1.5 bg-blue-600 text-white rounded-lg">Start</button>
                        )}
                        {t.status === 'In Progress' && (
                          <button onClick={() => { setCompleteModal({ id: t.id, title: t.title }); setCompleteNotes('') }}
                            className="text-xs px-3 py-1.5 bg-amber-500 text-white rounded-lg">Done</button>
                        )}
                        {t.status === 'Done' && <span className="text-xs text-amber-700">Awaiting HOD approval</span>}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Voice Add */}
          {canAssignTasks && (
          <div className="bg-gradient-to-r from-[#002B5B] to-blue-700 rounded-xl p-4 text-white">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="font-bold">🎙️ Assign by voice</h3>
                <p className="text-blue-200 text-xs mt-0.5">Responsibility: &quot;Vikash will enter delivery dates daily&quot; · Task: &quot;Complete warehouse audit by Friday&quot;</p>
              </div>
              <button onClick={startListening} disabled={isListening || aiParsing}
                className={`px-4 py-2 rounded-xl font-bold text-sm ${isListening ? 'bg-red-500 animate-pulse' : aiParsing ? 'bg-yellow-500' : 'bg-white text-[#002B5B] hover:bg-blue-50'} disabled:cursor-wait`}>
                {isListening ? '🔴 Listening…' : aiParsing ? '⏳ Parsing…' : '🎙️ Speak'}
              </button>
            </div>
            {voiceText && (
              <div className="bg-white/10 rounded-lg px-3 py-2 text-sm mb-2">
                <p className="text-blue-200 text-xs">Heard:</p>
                <p className="text-white font-medium">"{voiceText}"</p>
              </div>
            )}
            {aiParsed && (
              <div className="bg-white rounded-lg p-3 text-gray-800 text-sm space-y-2">
                <p className="font-semibold text-[#002B5B] text-xs uppercase">✅ Parsed — please confirm:</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div><span className="text-gray-500">Item Type:</span> <b>{aiParsed.mode === 'task' ? 'Task' : 'Responsibility'}</b></div>
                  <div><span className="text-gray-500">Employee:</span> <b>{aiParsed.employee_name || '—'}</b></div>
                  <div className="col-span-2"><span className="text-gray-500">Title:</span> <b>{aiParsed.title || '—'}</b></div>
                  {aiParsed.mode === 'responsibility' ? (
                    <div><span className="text-gray-500">Frequency:</span> <b>{aiParsed.frequency}</b></div>
                  ) : (
                    <div><span className="text-gray-500">Due Date:</span> <b>{aiParsed.due_date || '—'}</b></div>
                  )}
                </div>
                {!aiParsed.employee_id && <p className="text-amber-600 text-xs">⚠️ Employee not matched — select below</p>}
                <div className="flex gap-2 flex-wrap">
                  <button onClick={() => submitAssign(quickResp)}
                    disabled={createRespMut.isPending || createOneTimeTaskMut.isPending || !quickResp.employee_id || !quickResp.title}
                    className="px-3 py-1.5 bg-green-600 text-white rounded-lg text-xs font-medium disabled:opacity-50">
                    {(createRespMut.isPending || createOneTimeTaskMut.isPending) ? 'Saving…' : '✅ Save'}
                  </button>
                  <button onClick={() => setShowQuickResp(!showQuickResp)} className="px-3 py-1.5 border border-gray-300 rounded-lg text-xs bg-white">✏️ Edit</button>
                  <button onClick={() => { setAiParsed(null); setVoiceText('') }} className="px-3 py-1.5 border border-gray-300 rounded-lg text-xs bg-white">✕</button>
                </div>
              </div>
            )}
          </div>
          )}

          {/* Quick form */}
          {canAssignTasks && showQuickResp && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">Assignment Details</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div><label className="text-xs text-gray-500">Item Type *</label>
                  <select value={quickResp.mode} onChange={e => setQuickResp(f => ({ ...f, mode: e.target.value as 'responsibility' | 'task' }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="responsibility">Responsibility (recurring)</option>
                    <option value="task">Task (one-time)</option>
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">Employee *</label>
                  <select value={quickResp.employee_id} onChange={e => setQuickResp(f => ({ ...f, employee_id: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select</option>
                    {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name} — {e.department_name || '—'}</option>)}
                  </select>
                </div>
                <div className="col-span-2"><label className="text-xs text-gray-500">Title *</label>
                  <input value={quickResp.title} onChange={e => setQuickResp(f => ({ ...f, title: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div className="col-span-2"><label className="text-xs text-gray-500">Description</label>
                  <input value={quickResp.description} onChange={e => setQuickResp(f => ({ ...f, description: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1" placeholder="Optional details" /></div>
                {quickResp.mode === 'responsibility' ? (
                  <>
                    <div><label className="text-xs text-gray-500">Frequency</label>
                      <select value={quickResp.frequency} onChange={e => setQuickResp(f => ({ ...f, frequency: e.target.value }))}
                        className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                        {FREQUENCIES.map(f => <option key={f}>{f}</option>)}
                      </select>
                    </div>
                    <div><label className="text-xs text-gray-500">Category</label>
                      <select value={quickResp.category} onChange={e => setQuickResp(f => ({ ...f, category: e.target.value }))}
                        className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                        {CATEGORIES.map(c => <option key={c}>{c}</option>)}
                      </select>
                    </div>
                    <div><label className="text-xs text-gray-500">Linked Person (supervisor / approver)</label>
                      <select value={quickResp.linked_to_employee_id} onChange={e => setQuickResp(f => ({ ...f, linked_to_employee_id: e.target.value }))}
                        className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                        <option value="">Self-complete</option>
                        {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
                      </select>
                    </div>
                  </>
                ) : (
                  <div><label className="text-xs text-gray-500">Due Date</label>
                    <input type="date" value={quickResp.due_date} onChange={e => setQuickResp(f => ({ ...f, due_date: e.target.value }))}
                      className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                )}
                <div><label className="text-xs text-gray-500">{t(lang, 'assignedBy')}</label>
                  <select value={quickResp.added_by} onChange={e => setQuickResp(f => ({ ...f, added_by: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select</option>
                    {assigneeOptions.map(n => <option key={n}>{n}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">{t(lang, 'priority')}</label>
                  <select value={quickResp.priority} onChange={e => setQuickResp(f => ({ ...f, priority: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    {PRIORITIES.map(pr => <option key={pr}>{pr}</option>)}
                  </select>
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => submitAssign(quickResp)}
                  disabled={createRespMut.isPending || createOneTimeTaskMut.isPending || !quickResp.employee_id || !quickResp.title}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">
                  {(createRespMut.isPending || createOneTimeTaskMut.isPending) ? 'Saving…' : '✅ Assign'}
                </button>
                <button onClick={() => setShowQuickResp(false)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
              </div>
            </div>
          )}

          {/* Departments */}
          <div className="bg-white rounded-xl border p-4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-semibold text-gray-700">Departments</h3>
              {canManageOrg && (
              <button onClick={() => setShowDeptForm(true)} className="px-3 py-1.5 bg-[#002B5B] text-white rounded-lg text-xs font-medium">+ Add</button>
              )}
            </div>
            {showDeptForm && (
              <div className="grid grid-cols-3 gap-3 mb-3 bg-blue-50 p-3 rounded-lg">
                {[['name','Dept Name *'],['hod_name','HOD Name'],['description','Description']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(deptForm as any)[k]} onChange={e => setDeptForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                ))}
                <div className="flex gap-2 items-end">
                  <button onClick={() => createDeptMut.mutate(deptForm)} disabled={!deptForm.name} className="px-3 py-1.5 bg-[#002B5B] text-white rounded text-sm">Save</button>
                  <button onClick={() => setShowDeptForm(false)} className="px-3 py-1.5 border rounded text-sm">Cancel</button>
                </div>
              </div>
            )}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {(depts as any[]).map(d => (
                <div key={d.id} className="border rounded-lg p-3">
                  {editDept?.id === d.id ? (
                    <div className="space-y-2">
                      <input value={editDept.name} onChange={e => setEditDept((x: any) => ({ ...x, name: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm" placeholder="Dept name" />
                      <input value={editDept.hod_name} onChange={e => setEditDept((x: any) => ({ ...x, hod_name: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm" placeholder="HOD name" />
                      <div className="flex gap-1">
                        <button onClick={() => updateDeptMut.mutate({ id: d.id, data: editDept })} className="px-2 py-0.5 bg-green-600 text-white rounded text-xs">Save</button>
                        <button onClick={() => setEditDept(null)} className="px-2 py-0.5 border rounded text-xs">Cancel</button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <p className="font-semibold text-sm">{d.name}</p>
                      <p className="text-xs text-gray-500">HOD: {d.hod_name || '—'}</p>
                      {canManageOrg && (
                      <button onClick={() => setEditDept({ id: d.id, name: d.name, hod_name: d.hod_name || '' })} className="text-xs text-blue-600 mt-1">✏️ Edit</button>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── EMPLOYEE CHECK ── */}
      {tab === 'check' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            {!isEmployeeScope ? (
              <select
                value={checkEmp}
                onChange={e => setCheckEmp(e.target.value ? +e.target.value : '')}
                className="border rounded-lg px-3 py-1.5 text-sm min-w-[14rem]"
              >
                <option value="">Select Employee</option>
                {pickerEmps.map((e: any) => (
                  <option key={e.id} value={e.id}>{e.name} — {e.department_name || '—'}{e.designation ? ` · ${e.designation}` : ''}</option>
                ))}
              </select>
            ) : (
              <span className="text-sm font-medium text-gray-700">Your daily check</span>
            )}
            <input type="date" value={checkDate} onChange={e => setCheckDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
            <select value={checkPeriod} onChange={e => setCheckPeriod(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm">
              <option value="">{t(lang, 'timePeriod')} — All</option>
              {TIME_PERIODS.map(tp => <option key={tp}>{tp}</option>)}
            </select>
            <button onClick={() => setCheckDate(today())} className="text-xs px-2 py-1.5 border rounded-lg text-gray-600">Today</button>
            {canEditAssignments && checkEmp && (dayCheck?.summary?.unmarked_daily || 0) > 0 && (
              <button
                onClick={() => {
                  if (window.confirm(`Mark ${dayCheck?.summary.unmarked_daily} unmarked Daily item(s) as Missed for ${checkDate}?`)) {
                    markMissedMut.mutate()
                  }
                }}
                disabled={markMissedMut.isPending}
                className="text-xs px-3 py-1.5 bg-red-600 text-white rounded-lg disabled:opacity-50"
              >
                {markMissedMut.isPending ? 'Closing…' : `Auto-close unmarked as Missed (${dayCheck?.summary.unmarked_daily})`}
              </button>
            )}
            {canEditAssignments && checkEmp && (
              <button
                onClick={() => setShowReassign(true)}
                className="text-xs px-3 py-1.5 border border-amber-700 text-amber-900 rounded-lg"
              >
                Reassign mandatory (1 day)
              </button>
            )}
            <button onClick={() => setShowDailyGuide(v => !v)} className="text-xs px-3 py-1.5 border border-teal-700 text-teal-800 rounded-lg ml-auto">
              {showDailyGuide ? 'Hide daily guide' : 'Daily guide (Harsh / Sanjay)'}
            </button>
          </div>

          {showReassign && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 space-y-2 text-sm">
              <h4 className="font-semibold text-amber-900">One-day mandatory reassignment</h4>
              <p className="text-xs text-amber-800">Creates a temporary clone for the selected day under Additional Work. Original responsibility is not transferred permanently.</p>
              <div className="grid md:grid-cols-3 gap-2">
                <div>
                  <label className="text-xs text-gray-600">Responsibility ID</label>
                  <input
                    value={reassignForm.original_responsibility_id}
                    onChange={e => setReassignForm(f => ({ ...f, original_responsibility_id: e.target.value }))}
                    placeholder="From employee check list"
                    className="w-full border rounded px-2 py-1.5 text-sm mt-0.5"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-600">Assign to</label>
                  <select
                    value={reassignForm.to_employee_id}
                    onChange={e => setReassignForm(f => ({ ...f, to_employee_id: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-0.5"
                  >
                    <option value="">Select</option>
                    {pickerEmps.map((e: any) => (
                      <option key={e.id} value={e.id}>{e.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-600">Date</label>
                  <input type="date" value={reassignForm.reassignment_date} onChange={e => setReassignForm(f => ({ ...f, reassignment_date: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-0.5" />
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={!reassignForm.original_responsibility_id || !reassignForm.to_employee_id || reassignMut.isPending}
                  onClick={() => reassignMut.mutate({
                    original_responsibility_id: +reassignForm.original_responsibility_id,
                    to_employee_id: +reassignForm.to_employee_id,
                    reassignment_date: reassignForm.reassignment_date,
                  })}
                  className="px-3 py-1.5 bg-amber-800 text-white rounded-lg text-xs disabled:opacity-50"
                >
                  Create one-day clone
                </button>
                <button type="button" onClick={() => setShowReassign(false)} className="px-3 py-1.5 border rounded-lg text-xs">Cancel</button>
              </div>
            </div>
          )}

          {showDailyGuide && (
            <div className="bg-slate-50 border rounded-xl p-4 text-sm space-y-4">
              <div>
                <h4 className="font-semibold text-[#002B5B]">Harsh (IT Admin) — daily</h4>
                <ol className="list-decimal ml-5 mt-1 space-y-1 text-gray-700 text-xs">
                  <li>Open HRM → Employee Check (auto-loads you if logged in as Harsh).</li>
                  <li>Morning: complete systems check → mark <b>Done</b> on HOD View / ask Admin to open your dept grid.</li>
                  <li>Through the day: update IT tickets + HRM task statuses; mark SLA + preventive maintenance.</li>
                  <li>Before leaving: EOD report Done; confirm nothing left Pending on Check tab.</li>
                  <li>Weekly: documentation. Monthly: one improvement initiative.</li>
                </ol>
              </div>
              <div>
                <h4 className="font-semibold text-[#002B5B]">Sanjay (Sales Head) — daily</h4>
                <ol className="list-decimal ml-5 mt-1 space-y-1 text-gray-700 text-xs">
                  <li>Open HRM → Employee Check for Sanjay Sodani.</li>
                  <li>10:00–10:30: sales vs target → mark Sales performance monitoring Done.</li>
                  <li>10:30: team meeting → mark Team management Done.</li>
                  <li>Marketplace + growth blocks → mark those Done/Partial with notes if needed.</li>
                  <li>Escalation-only coordination: mark Done only if you stayed out of ops work (or Partial/Blocked with reason).</li>
                  <li>Before leaving: submit outcome DSR → mark DSR Done. Weekly: promotional initiative.</li>
                </ol>
              </div>
              <p className="text-xs text-gray-500">Manager: select employee + date here to see Worked on vs Not worked. Use Auto-close at day end for leftover Pending Dailies.</p>
            </div>
          )}

          {!checkEmp && <p className="text-center text-gray-400 py-8 text-sm">Select an employee to see what they worked on and what they did not.</p>}
          {checkEmp && dayCheckLoading && !dayCheck && <p className="text-center text-gray-400 py-8 text-sm">Loading…</p>}
          {dayCheck && (
            <div className="space-y-4">
              <div className="bg-teal-800 text-white rounded-xl p-4 flex flex-wrap justify-between gap-3">
                <div>
                  <h3 className="font-bold text-lg">{dayCheck.employee?.name}</h3>
                  <p className="text-teal-100 text-sm">{dayCheck.employee?.department_name} · {dayCheck.employee?.designation || '—'}</p>
                  <p className="text-teal-200 text-xs mt-1">Check date: {dayCheck.check_date} · this screen is the Daily Work Report (DWR)</p>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-bold">{dayCheck?.summary?.completion_pct ?? 0}%</p>
                  <p className="text-teal-200 text-xs">Daily completion</p>
                  <p className="text-teal-100 text-xs mt-1">
                    Done {dayCheck?.summary?.daily_done ?? 0} · Partial {dayCheck?.summary?.daily_partial ?? 0} · Pending {dayCheck?.summary?.daily_pending ?? 0} · Missed {dayCheck?.summary?.daily_missed ?? 0}
                  </p>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white rounded-xl border overflow-hidden">
                  <div className="px-4 py-2.5 bg-green-600 text-white font-semibold text-sm">Worked on ({dayCheckFiltered?.worked_on?.length || 0})</div>
                  {(dayCheckFiltered?.worked_on || []).length === 0 ? (
                    <p className="p-4 text-sm text-gray-400">No Done/Partial marks for this date yet.</p>
                  ) : (
                    <ul className="divide-y">
                      {(dayCheckFiltered?.worked_on || []).map((i: any) => (
                        <li key={i.responsibility_id} className="px-4 py-3">
                          <div className="flex items-start gap-2">
                            <span className={`mt-0.5 inline-flex w-6 h-6 items-center justify-center rounded-full text-xs ${statusBg(i.status)}`}>{statusLabel(i.status)}</span>
                            {renderRespCheck(i, { showMark: canEditAssignments })}
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                <div className="bg-white rounded-xl border overflow-hidden">
                  <div className="px-4 py-2.5 bg-red-600 text-white font-semibold text-sm">Not worked / pending ({dayCheckFiltered?.not_worked?.length || 0})</div>
                  {(dayCheckFiltered?.not_worked || []).length === 0 ? (
                    <p className="p-4 text-sm text-gray-400">Nothing pending or missed — good.</p>
                  ) : (
                    <ul className="divide-y">
                      {(dayCheckFiltered?.not_worked || []).map((i: any) => (
                        <li key={i.responsibility_id} className="px-4 py-3">
                          <div className="flex items-start gap-2">
                            <span className={`mt-0.5 inline-flex w-6 h-6 items-center justify-center rounded-full text-xs ${statusBg(i.status)}`}>{statusLabel(i.status)}</span>
                            {renderRespCheck(i)}
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>

              {((dayCheckFiltered?.additional_work || []).length > 0) && (
                <div className="bg-white rounded-xl border overflow-hidden">
                  <div className="px-4 py-2.5 bg-amber-700 text-white font-semibold text-sm">
                    Additional Work (Assigned by HOD) ({dayCheckFiltered?.additional_work?.length || 0})
                  </div>
                  <ul className="divide-y">
                    {(dayCheckFiltered?.additional_work || []).map((i: any) => (
                      <li key={i.clone_id} className="px-4 py-3 flex items-start gap-2">
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-sm text-gray-800">{i.title}</p>
                          <p className="text-xs text-gray-400">Cover for {i.original_employee_name} · {i.status}{i.assigned_by ? ` · by ${i.assigned_by}` : ''}</p>
                          {i.status === 'Pending' && (
                            <select
                              defaultValue=""
                              onChange={e => {
                                const val = e.target.value
                                if (!val) return
                                markCloneMut.mutate({ id: i.clone_id, status: val })
                                e.target.value = ''
                              }}
                              className="mt-1.5 text-xs border rounded px-1.5 py-1 bg-white"
                            >
                              <option value="">Mark status…</option>
                              {TASK_LOG_STATUSES.map(st => <option key={st} value={st}>{st}</option>)}
                            </select>
                          )}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {((dayCheckFiltered?.whenever_required || []).length > 0) && (
                <div className="bg-white rounded-xl border overflow-hidden">
                  <div className="px-4 py-2.5 bg-slate-700 text-white font-semibold text-sm">
                    Daily Tasks — Whenever Required ({dayCheckFiltered?.whenever_required?.length || 0})
                  </div>
                  <p className="px-4 pt-2 text-[11px] text-gray-500">Use N/A when not required (no performance impact). Mark Done/Partial when actually performed.</p>
                  <ul className="divide-y">
                    {(dayCheckFiltered?.whenever_required || []).map((i: any) => (
                      <li key={i.responsibility_id} className="px-4 py-3">
                        <div className="flex items-start gap-2">
                          <span className={`mt-0.5 inline-flex w-6 h-6 items-center justify-center rounded-full text-xs ${statusBg(i.status)}`}>{statusLabel(i.status)}</span>
                          {renderRespCheck(i)}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {((dayCheckFiltered?.one_time_working || []).length > 0 || (dayCheckFiltered?.one_time_pending || []).length > 0 || (dayCheckFiltered?.one_time_awaiting_approval || []).length > 0) && (
                <div className="bg-white rounded-xl border p-4 space-y-3">
                  <h4 className="font-semibold text-[#002B5B] text-sm">One-time tasks</h4>
                  {(dayCheckFiltered?.one_time_working || []).map((t: any) => (
                    <p key={t.id} className="text-sm"><span className="text-blue-700 font-medium">In progress:</span> {t.title}</p>
                  ))}
                  {(dayCheckFiltered?.one_time_pending || []).map((t: any) => (
                    <p key={t.id} className="text-sm"><span className="text-gray-600 font-medium">{t.status}:</span> {t.title}{t.due_date ? ` (due ${t.due_date})` : ''}</p>
                  ))}
                  {(dayCheckFiltered?.one_time_awaiting_approval || []).map((t: any) => (
                    <p key={t.id} className="text-sm"><span className="text-amber-700 font-medium">Awaiting HOD:</span> {t.title}</p>
                  ))}
                </div>
              )}

              <p className="text-xs text-gray-500">
                To mark statuses: open <button type="button" className="underline text-teal-800" onClick={() => {
                  const emp = (allEmps as any[]).find((e: any) => e.id === checkEmp)
                  if (emp?.department_id) { setHodDept(emp.department_id); setHodEmp(checkEmp as number); setFromDate(checkDate); setToDate(checkDate); setTab('hod') }
                }}>HOD View</button> for this employee&apos;s department, or use Appraisal for the period score.
              </p>
            </div>
          )}
        </div>
      )}

      {/* ── EMPLOYEES ── */}
      {tab === 'employees' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <select
              value={selDept}
              onChange={e => setSelDept(e.target.value ? +e.target.value : '')}
              disabled={scopeLevel === 'department'}
              className="border rounded-lg px-3 py-1.5 text-sm disabled:bg-gray-100"
            >
              <option value="">All Departments</option>
              {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
            {(canManageOrg || scopeLevel === 'department') && (
            <button onClick={() => setShowEmpForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium">+ Add Employee</button>
            )}
          </div>
          {showEmpForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Employee</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div className="relative"><label className="text-xs text-gray-500">Name *</label>
                  <input value={empForm.name} onChange={e => searchEmpNames(e.target.value)} className="w-full border rounded px-2 py-1.5 text-sm mt-1" autoComplete="off" />
                  {empNameSuggest.length > 0 && (
                    <ul className="absolute z-10 bg-white border rounded shadow-sm mt-0.5 w-full max-h-32 overflow-auto text-xs">
                      {empNameSuggest.map((s: any) => (
                        <li key={s.id} className="px-2 py-1 text-amber-800 border-b last:border-0">⚠ Already exists: {s.name} ({s.emp_code})</li>
                      ))}
                    </ul>
                  )}
                </div>
                <div><label className="text-xs text-gray-500">Employee ID</label>
                  <input value={empForm.emp_code} onChange={e => setEmpForm(f => ({ ...f, emp_code: e.target.value }))} placeholder="Auto if blank" className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div><label className="text-xs text-gray-500">Department</label>
                  <select value={empForm.department_id} onChange={e => setEmpForm(f => ({ ...f, department_id: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select</option>
                    {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
                  </select>
                </div>
                {[['designation','Designation'],['phone','Phone'],['email','Email'],['join_date','Join Date']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input type={k === 'join_date' ? 'date' : 'text'} value={(empForm as any)[k]} onChange={e => setEmpForm(f => ({ ...f, [k]: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                ))}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createEmpMut.mutate({ ...empForm, department_id: empForm.department_id ? +empForm.department_id : null, emp_code: empForm.emp_code || undefined })} disabled={!empForm.name} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">Save</button>
                <button onClick={() => setShowEmpForm(false)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
              </div>
            </div>
          )}
          <div className="bg-white rounded-xl border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-xs text-gray-400 uppercase">
                <tr>{['Code','Name','Department','Designation','Email','Phone','Status',''].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {(employees as any[]).map((e: any) => (
                  <tr key={e.id} className="border-t hover:bg-gray-50">
                    {editEmp?.id === e.id ? (
                      <td colSpan={8} className="px-4 py-3">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                          <input value={editEmp.emp_code || ''} onChange={ev => setEditEmp((x: any) => ({ ...x, emp_code: ev.target.value }))} className="border rounded px-2 py-1 text-sm font-mono" placeholder="Emp ID" />
                          <input value={editEmp.name} onChange={ev => setEditEmp((x: any) => ({ ...x, name: ev.target.value }))} className="border rounded px-2 py-1 text-sm" placeholder="Name" />
                          <select value={editEmp.department_id || ''} onChange={ev => setEditEmp((x: any) => ({ ...x, department_id: +ev.target.value }))} className="border rounded px-2 py-1 text-sm">
                            <option value="">Dept</option>
                            {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
                          </select>
                          <input value={editEmp.designation} onChange={ev => setEditEmp((x: any) => ({ ...x, designation: ev.target.value }))} className="border rounded px-2 py-1 text-sm" placeholder="Designation" />
                          <input value={editEmp.email || ''} onChange={ev => setEditEmp((x: any) => ({ ...x, email: ev.target.value }))} className="border rounded px-2 py-1 text-sm" placeholder="Email" />
                          <div className="flex gap-1 col-span-2">
                            <button onClick={() => updateEmpMut.mutate({ id: e.id, data: { name: editEmp.name, emp_code: editEmp.emp_code, department_id: editEmp.department_id, designation: editEmp.designation, email: editEmp.email } })} className="px-2 py-1 bg-green-600 text-white rounded text-xs">Save</button>
                            <button onClick={() => setEditEmp(null)} className="px-2 py-1 border rounded text-xs">Cancel</button>
                          </div>
                        </div>
                      </td>
                    ) : (
                      <>
                        <td className="px-4 py-2 font-mono text-xs text-[#002B5B]">{e.emp_code}</td>
                        <td className="px-4 py-2 font-semibold">{e.name}</td>
                        <td className="px-4 py-2 text-gray-500">{e.department_name || '—'}</td>
                        <td className="px-4 py-2 text-gray-500">{e.designation || '—'}</td>
                        <td className="px-4 py-2 text-gray-500">{e.email || '—'}</td>
                        <td className="px-4 py-2 text-gray-500">{e.phone || '—'}</td>
                        <td className="px-4 py-2"><span className={`text-xs px-2 py-0.5 rounded-full ${e.status === 'Active' ? 'bg-green-100 text-green-700' : 'bg-gray-100'}`}>{e.status}</span></td>
                        <td className="px-4 py-2">
                          <div className="flex gap-2">
                            <button onClick={() => setEditEmp({ id: e.id, emp_code: e.emp_code, name: e.name, department_id: e.department_id, designation: e.designation || '', email: e.email || '' })} className="text-xs text-blue-600">✏️</button>
                            {canViewEmployeeList && (
                              <button onClick={() => { if (window.confirm(`Delete employee ${e.name} (${e.emp_code})?`)) deleteEmpMut.mutate(e.id) }} className="text-xs text-red-600">🗑️</button>
                            )}
                            <button onClick={() => { setCheckEmp(e.id); setCheckDate(today()); setTab('check') }} className="text-xs text-teal-700">🔎 Check</button>
                            <button onClick={() => { setAppraisalEmp(e.id); setTab('appraisal') }} className="text-xs text-purple-600">📁 Appraisal</button>
                          </div>
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
            {(employees as any[]).length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No employees.</p>}
          </div>
        </div>
      )}

      {/* ── RESPONSIBILITIES ── */}
      {tab === 'responsibilities' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex gap-2">
              {!isEmployeeScope && (
                <>
                  <select value={selDept} onChange={e => setSelDept(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
                    <option value="">All Departments</option>
                    {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
                  </select>
                  <select value={selEmp} onChange={e => setSelEmp(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
                    <option value="">All Employees</option>
                    {pickerEmps.map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
                  </select>
                </>
              )}
            </div>
            <div className="flex gap-2 flex-wrap">
            {(canEditAssignments || isEmployeeScope) && (
            <button onClick={() => setShowRespForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium">
              {isEmployeeScope ? '+ Add My Item' : '+ Assign Item'}
            </button>
            )}
            {canEditAssignments && (
              <>
                <input ref={respImportRef} type="file" accept=".csv,.xlsx,.xls" className="hidden"
                  onChange={e => { const f = e.target.files?.[0]; if (f) importRespMut.mutate(f); e.target.value = '' }} />
                <button onClick={() => respImportRef.current?.click()} disabled={importRespMut.isPending}
                  className="px-4 py-2 border border-[#002B5B] text-[#002B5B] rounded-lg text-sm font-medium disabled:opacity-50">
                  {importRespMut.isPending ? 'Importing…' : '📥 Import Sheet'}
                </button>
              </>
            )}
            </div>
          </div>
          {showRespForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">Assign Responsibility</h3>
              <p className="text-[11px] text-gray-500">{t(lang, 'scheduleNote')}</p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div><label className="text-xs text-gray-500">Employee *</label>
                  <select value={respForm.employee_id} onChange={e => setRespForm(f => ({ ...f, employee_id: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select</option>
                    {pickerEmps.map((e: any) => <option key={e.id} value={e.id}>{e.name} ({e.department_name || '—'})</option>)}
                  </select>
                </div>
                <div className="col-span-2"><label className="text-xs text-gray-500">Title *</label>
                  <input value={respForm.title} onChange={e => setRespForm(f => ({ ...f, title: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div className="col-span-2"><label className="text-xs text-gray-500">Description</label>
                  <input value={respForm.description} onChange={e => setRespForm(f => ({ ...f, description: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div><label className="text-xs text-gray-500">{t(lang, 'frequency')}</label>
                  <select value={respForm.frequency} onChange={e => setRespForm(f => ({ ...f, frequency: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    {FREQUENCIES.map(f => <option key={f}>{f}</option>)}
                  </select>
                </div>
                {(respForm.frequency === 'Weekly' || respForm.frequency === 'Fortnightly') && (
                  <div><label className="text-xs text-gray-500">{t(lang, 'weekday')} *</label>
                    <select value={respForm.schedule_weekday} onChange={e => setRespForm(f => ({ ...f, schedule_weekday: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                      <option value="">Select</option>
                      {WEEKDAYS.map(d => <option key={d}>{d}</option>)}
                    </select>
                  </div>
                )}
                {respForm.frequency === 'Monthly' && (
                  <div><label className="text-xs text-gray-500">{t(lang, 'monthDay')} *</label>
                    <input type="number" min={1} max={31} value={respForm.schedule_month_day || ''} onChange={e => setRespForm(f => ({ ...f, schedule_month_day: +e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                )}
                {respForm.frequency === 'Quarterly' && (
                  <div><label className="text-xs text-gray-500">Anchor month *</label>
                    <select value={respForm.schedule_month || ''} onChange={e => setRespForm(f => ({ ...f, schedule_month: +e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                      <option value="">Select month</option>
                      {MONTHS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                    </select>
                    <p className="text-[10px] text-gray-400 mt-0.5">Repeats every 3 months from the selected month</p>
                  </div>
                )}
                <div><label className="text-xs text-gray-500">Linked Person (supervisor / approver)</label>
                  <select value={respForm.linked_to_employee_id} onChange={e => setRespForm(f => ({ ...f, linked_to_employee_id: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Self-complete (no approval)</option>
                    {pickerEmps.map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
                  </select>
                  <p className="text-[10px] text-gray-400 mt-0.5">Shown on Employee Check so the employee knows who supervises or approves this task.</p>
                </div>
                <div><label className="text-xs text-gray-500">Category</label>
                  <select value={respForm.category} onChange={e => setRespForm(f => ({ ...f, category: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    {CATEGORIES.map(c => <option key={c}>{c}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">{t(lang, 'priority')}</label>
                  <select value={respForm.priority} onChange={e => setRespForm(f => ({ ...f, priority: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    {PRIORITIES.map(pr => <option key={pr}>{pr}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">{t(lang, 'timePeriod')}</label>
                  <select value={respForm.time_period} onChange={e => setRespForm(f => ({ ...f, time_period: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">—</option>
                    {TIME_PERIODS.map(tp => <option key={tp}>{tp}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">{t(lang, 'mandatory')}</label>
                  <select value={respForm.mandatory ? 'yes' : 'no'} onChange={e => setRespForm(f => ({ ...f, mandatory: e.target.value === 'yes' }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="no">{t(lang, 'no')}</option>
                    <option value="yes">{t(lang, 'yes')}</option>
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">{t(lang, 'assignedBy')}</label>
                  <select value={respForm.added_by} onChange={e => setRespForm(f => ({ ...f, added_by: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select</option>
                    {assigneeOptions.map(n => <option key={n}>{n}</option>)}
                  </select>
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => submitAssign({ ...respForm, mode: 'responsibility' as const, department_id: '', due_date: '' })} disabled={!respForm.employee_id || !respForm.title || createRespMut.isPending} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">Assign</button>
                <button onClick={() => setShowRespForm(false)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
              </div>
            </div>
          )}
          {(() => {
            const grouped: Record<string, any[]> = {}
            ;filteredResponsibilities.forEach((r: any) => { const key = r.employee_name || 'Unknown'; if (!grouped[key]) grouped[key] = []; grouped[key].push(r) })
            return (
              <>
                <div className="bg-white rounded-xl border p-3 flex flex-wrap gap-2 text-xs">
                  <span className="font-semibold text-gray-500 self-center">{t(lang, 'filters')}:</span>
                  <input placeholder="Title" value={respTitleFilter} onChange={e => setRespTitleFilter(e.target.value)} className="border rounded px-2 py-1" />
                  <select value={respFreqFilter} onChange={e => setRespFreqFilter(e.target.value)} className="border rounded px-2 py-1">
                    <option value="">Frequency</option>
                    {FREQUENCIES.map(f => <option key={f}>{f}</option>)}
                  </select>
                  <select value={respPriorityFilter} onChange={e => setRespPriorityFilter(e.target.value)} className="border rounded px-2 py-1">
                    <option value="">Priority</option>
                    {PRIORITIES.map(p => <option key={p}>{p}</option>)}
                  </select>
                  <input placeholder="Assigned By" value={respAssignedByFilter} onChange={e => setRespAssignedByFilter(e.target.value)} className="border rounded px-2 py-1" />
                </div>
            {Object.entries(grouped).map(([empName, resps]) => (
              <div key={empName} className="bg-white rounded-xl border overflow-hidden">
                <div className="px-4 py-2 bg-[#002B5B] text-white text-sm font-semibold flex justify-between">
                  <span>👤 {empName}</span>
                  <span className="text-blue-200 text-xs">{resps[0]?.department_name || ''} · {resps.length} tasks</span>
                </div>
                <table className="w-full text-sm">
                  <thead className="text-gray-400 text-xs uppercase bg-gray-50">
                    <tr><th className="text-left px-4 py-2">Task</th><th className="text-left px-4 py-2">Description</th><th className="text-left px-4 py-2">Frequency</th><th className="text-left px-4 py-2">Linked Person</th><th className="text-left px-4 py-2">Priority</th><th className="text-left px-4 py-2">Added By</th><th className="px-4 py-2"></th></tr>
                  </thead>
                  <tbody>
                    {resps.map((r: any) => (
                      <tr key={r.id} className="border-t hover:bg-gray-50">
                        {editResp?.id === r.id ? (
                          <td colSpan={7} className="px-4 py-3">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                              <input value={editResp.title} onChange={e => setEditResp((x: any) => ({ ...x, title: e.target.value }))} className="border rounded px-2 py-1 text-sm col-span-2" placeholder="Title" />
                              <input value={editResp.description || ''} onChange={e => setEditResp((x: any) => ({ ...x, description: e.target.value }))} className="border rounded px-2 py-1 text-sm col-span-2" placeholder="Description" />
                              <div>
                                <label className="text-[10px] text-gray-400">Frequency</label>
                                <select value={editResp.frequency} onChange={e => setEditResp((x: any) => ({ ...x, frequency: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm">
                                  {FREQUENCIES.map(f => <option key={f}>{f}</option>)}
                                </select>
                              </div>
                              {(editResp.frequency === 'Weekly' || editResp.frequency === 'Fortnightly') && (
                                <div>
                                  <label className="text-[10px] text-gray-400">Weekday *</label>
                                  <select value={editResp.schedule_weekday || ''} onChange={e => setEditResp((x: any) => ({ ...x, schedule_weekday: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm">
                                    <option value="">Select</option>
                                    {WEEKDAYS.map(d => <option key={d}>{d}</option>)}
                                  </select>
                                </div>
                              )}
                              {editResp.frequency === 'Monthly' && (
                                <div>
                                  <label className="text-[10px] text-gray-400">Day of month *</label>
                                  <input type="number" min={1} max={31} value={editResp.schedule_month_day || ''} onChange={e => setEditResp((x: any) => ({ ...x, schedule_month_day: +e.target.value }))} className="w-full border rounded px-2 py-1 text-sm" />
                                </div>
                              )}
                              {editResp.frequency === 'Quarterly' && (
                                <div>
                                  <label className="text-[10px] text-gray-400">Anchor month *</label>
                                  <select value={editResp.schedule_month || ''} onChange={e => setEditResp((x: any) => ({ ...x, schedule_month: +e.target.value }))} className="w-full border rounded px-2 py-1 text-sm">
                                    <option value="">Select</option>
                                    {MONTHS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                                  </select>
                                </div>
                              )}
                              <div>
                                <label className="text-[10px] text-gray-400">Category</label>
                                <select value={editResp.category} onChange={e => setEditResp((x: any) => ({ ...x, category: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm">
                                  {CATEGORIES.map(c => <option key={c}>{c}</option>)}
                                </select>
                              </div>
                              <div>
                                <label className="text-[10px] text-gray-400">Priority</label>
                                <select value={editResp.priority || 'Medium'} onChange={e => setEditResp((x: any) => ({ ...x, priority: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm">
                                  {PRIORITIES.map(pr => <option key={pr}>{pr}</option>)}
                                </select>
                              </div>
                              <div>
                                <label className="text-[10px] text-gray-400">Mandatory</label>
                                <select value={editResp.mandatory ? 'yes' : 'no'} onChange={e => setEditResp((x: any) => ({ ...x, mandatory: e.target.value === 'yes' }))} className="w-full border rounded px-2 py-1 text-sm">
                                  <option value="no">No</option>
                                  <option value="yes">Yes</option>
                                </select>
                              </div>
                              <div>
                                <label className="text-[10px] text-gray-400">Time Period</label>
                                <select value={editResp.time_period || ''} onChange={e => setEditResp((x: any) => ({ ...x, time_period: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm">
                                  <option value="">—</option>
                                  {TIME_PERIODS.map(tp => <option key={tp}>{tp}</option>)}
                                </select>
                              </div>
                              <div className="col-span-2">
                                <label className="text-[10px] text-gray-400">Assigned To</label>
                                <select value={editResp.employee_id} onChange={e => setEditResp((x: any) => ({ ...x, employee_id: +e.target.value }))} className="w-full border rounded px-2 py-1 text-sm">
                                  {(canEditAssignments ? pickerEmps : allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
                                </select>
                              </div>
                              <div className="col-span-2">
                                <label className="text-[10px] text-gray-400">Linked To (supervisor / approver)</label>
                                <select value={editResp.linked_to_employee_id || ''} onChange={e => setEditResp((x: any) => ({ ...x, linked_to_employee_id: e.target.value ? +e.target.value : '' }))} className="w-full border rounded px-2 py-1 text-sm">
                                  <option value="">Self-complete (no Linked Person)</option>
                                  {(canEditAssignments ? pickerEmps : allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
                                </select>
                              </div>
                              <div className="flex gap-2 col-span-2">
                                <button onClick={() => {
                                  if ((editResp.frequency === 'Weekly' || editResp.frequency === 'Fortnightly') && !editResp.schedule_weekday) { alert('Select a weekday for Weekly/Fortnightly'); return }
                                  if (editResp.frequency === 'Monthly' && !(editResp.schedule_month_day > 0)) { alert('Select calendar day for Monthly'); return }
                                  if (editResp.frequency === 'Quarterly' && !(editResp.schedule_month > 0)) { alert('Select anchor month for Quarterly'); return }
                                  updateRespMut.mutate({ id: r.id, data: { title: editResp.title, description: editResp.description, frequency: editResp.frequency, category: editResp.category, employee_id: editResp.employee_id, linked_to_employee_id: editResp.linked_to_employee_id || null, priority: editResp.priority || 'Medium', mandatory: !!editResp.mandatory, schedule_weekday: editResp.schedule_weekday || '', schedule_month_day: editResp.schedule_month_day || 0, schedule_month: editResp.schedule_month || 0, time_period: editResp.time_period || '' } })
                                }} disabled={!editResp.title || updateRespMut.isPending} className="px-3 py-1 bg-green-600 text-white rounded text-xs">Save</button>
                                <button onClick={() => setEditResp(null)} className="px-3 py-1 border rounded text-xs">Cancel</button>
                              </div>
                            </div>
                          </td>
                        ) : (
                          <>
                            <td className="px-4 py-2 font-medium">{r.title}{r.mandatory ? <span className="ml-1 text-[10px] text-red-600">*</span> : null}</td>
                            <td className="px-4 py-2 text-xs text-gray-500">{r.description || '—'}</td>
                            <td className="px-4 py-2"><span className={`text-xs px-2 py-0.5 rounded-full font-medium ${r.frequency === 'Daily' ? 'bg-blue-100 text-blue-700' : r.frequency === 'Weekly' ? 'bg-purple-100 text-purple-700' : 'bg-green-100 text-green-700'}`}>{r.frequency}{r.schedule_weekday ? ` · ${r.schedule_weekday}` : ''}{r.schedule_month_day ? ` · D${r.schedule_month_day}` : ''}</span></td>
                            <td className="px-4 py-2 text-xs text-indigo-800">{r.linked_to_employee_name || 'Self-complete'}</td>
                            <td className="px-4 py-2 text-xs"><span className={`px-1.5 py-0.5 rounded ${priorityStyle(r.priority || 'Medium')}`}>{r.priority || 'Medium'}</span></td>
                            <td className="px-4 py-2 text-xs text-gray-400">{r.added_by || '—'}</td>
                            <td className="px-4 py-2">
                              {canMutateRecords && (
                                <div className="flex gap-2">
                                  <button onClick={() => setEditResp({ id: r.id, title: r.title, description: r.description || '', frequency: r.frequency, category: r.category, employee_id: r.employee_id, linked_to_employee_id: r.linked_to_employee_id || '', priority: r.priority || 'Medium', mandatory: !!r.mandatory, schedule_weekday: r.schedule_weekday || '', schedule_month_day: r.schedule_month_day || 0, schedule_month: r.schedule_month || 0, time_period: r.time_period || '' })} className="text-xs text-blue-600">✏️</button>
                                  {canDeleteHrm && (
                                    <button onClick={() => { if (window.confirm('Remove?')) deleteRespMut.mutate(r.id) }} className="text-xs text-red-500">🗑️</button>
                                  )}
                                </div>
                              )}
                            </td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
              </>
            )
          })()}
        </div>
      )}

      {/* ── ONE-TIME TASKS ── */}
      {tab === 'tasks' && (
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-100 rounded-xl px-4 py-3 text-sm text-blue-900">
            <b>Tasks</b> are one-time assignments with time tracking and HOD approval.
            Recurring daily/weekly work stays under <b>Responsibilities</b> and the HOD view.
          </div>
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex gap-2 flex-wrap">
              {!isEmployeeScope && (
                <select value={selDept} onChange={e => setSelDept(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
                  <option value="">All Departments</option>
                  {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
                </select>
              )}
              {!isEmployeeScope && (
                <select value={selEmp} onChange={e => setSelEmp(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
                  <option value="">All Employees</option>
                  {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
                </select>
              )}
              <select value={taskStatusFilter} onChange={e => setTaskStatusFilter(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Statuses</option>
                {ONE_TIME_STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            {canAssignTasks && (
              <div className="flex gap-2 flex-wrap">
                <button onClick={() => setShowTaskForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium">+ Assign Item</button>
                <input ref={taskImportRef} type="file" accept=".csv,.xlsx,.xls" className="hidden"
                  onChange={e => { const f = e.target.files?.[0]; if (f) importTaskMut.mutate(f); e.target.value = '' }} />
                <button onClick={() => taskImportRef.current?.click()} disabled={importTaskMut.isPending}
                  className="px-4 py-2 border border-[#002B5B] text-[#002B5B] rounded-lg text-sm font-medium disabled:opacity-50">
                  {importTaskMut.isPending ? 'Importing…' : '📥 Import Sheet'}
                </button>
              </div>
            )}
          </div>

          {showTaskForm && canAssignTasks && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">Assign Task</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div><label className="text-xs text-gray-500">Employee *</label>
                  <select value={taskForm.employee_id} onChange={e => setTaskForm(f => ({ ...f, employee_id: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select</option>
                    {pickerEmps.map((e: any) => <option key={e.id} value={e.id}>{e.name} ({e.department_name || '—'})</option>)}
                  </select>
                </div>
                <div className="col-span-2"><label className="text-xs text-gray-500">Title *</label>
                  <input value={taskForm.title} onChange={e => setTaskForm(f => ({ ...f, title: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div className="col-span-2"><label className="text-xs text-gray-500">Description</label>
                  <input value={taskForm.description} onChange={e => setTaskForm(f => ({ ...f, description: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div><label className="text-xs text-gray-500">{t(lang, 'dueDate')}</label>
                  <input type="date" value={taskForm.due_date} onChange={e => setTaskForm(f => ({ ...f, due_date: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div><label className="text-xs text-gray-500">{t(lang, 'priority')}</label>
                  <select value={taskForm.priority} onChange={e => setTaskForm(f => ({ ...f, priority: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    {PRIORITIES.map(pr => <option key={pr}>{pr}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">{t(lang, 'assignedBy')}</label>
                  <select value={taskForm.assigned_by} onChange={e => setTaskForm(f => ({ ...f, assigned_by: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select</option>
                    {assigneeOptions.map(n => <option key={n}>{n}</option>)}
                  </select>
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => submitAssign({
                    mode: 'task',
                    employee_id: taskForm.employee_id,
                    department_id: '',
                    title: taskForm.title,
                    description: taskForm.description,
                    frequency: 'Daily',
                    category: 'General',
                    added_by: taskForm.assigned_by,
                    due_date: taskForm.due_date,
                    priority: taskForm.priority,
                    mandatory: false,
                    schedule_weekday: '',
                    schedule_month_day: 0,
                    schedule_month: 0,
                    time_period: '',
                    linked_to_employee_id: '',
                  })}
                  disabled={!taskForm.employee_id || !taskForm.title || createOneTimeTaskMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">
                  Assign
                </button>
                <button onClick={() => setShowTaskForm(false)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
              </div>
            </div>
          )}

          <div className="bg-white rounded-xl border overflow-hidden">
            <div className="px-3 py-2 flex flex-wrap gap-2 text-xs border-b">
              <input placeholder="Title" value={taskTitleFilter} onChange={e => setTaskTitleFilter(e.target.value)} className="border rounded px-2 py-1" />
              <select value={taskPriorityFilter} onChange={e => setTaskPriorityFilter(e.target.value)} className="border rounded px-2 py-1">
                <option value="">Priority</option>
                {PRIORITIES.map(p => <option key={p}>{p}</option>)}
              </select>
              <input placeholder="Assigned By" value={taskAssignedByFilter} onChange={e => setTaskAssignedByFilter(e.target.value)} className="border rounded px-2 py-1" />
            </div>
            <table className="w-full text-sm">
              <thead className="text-gray-400 text-xs uppercase bg-gray-50">
                <tr>
                  <th className="text-left px-4 py-2">Task</th>
                  <th className="text-left px-4 py-2">Employee</th>
                  <th className="text-left px-4 py-2">Assigned By</th>
                  <th className="text-left px-4 py-2">Priority</th>
                  <th className="text-left px-4 py-2">Due</th>
                  <th className="text-left px-4 py-2">Status</th>
                  <th className="text-left px-4 py-2">Time</th>
                  <th className="px-4 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredTasks.map((t: any) => (
                  <tr key={t.id} className="border-t hover:bg-gray-50 align-top">
                    {editTask?.id === t.id ? (
                      <td colSpan={8} className="px-4 py-3">
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                          <input value={editTask.title} onChange={e => setEditTask((x: any) => ({ ...x, title: e.target.value }))} className="border rounded px-2 py-1 text-sm col-span-2" placeholder="Task title" />
                          <input value={editTask.description || ''} onChange={e => setEditTask((x: any) => ({ ...x, description: e.target.value }))} className="border rounded px-2 py-1 text-sm col-span-2" placeholder="Description" />
                          <input type="date" value={editTask.due_date || ''} onChange={e => setEditTask((x: any) => ({ ...x, due_date: e.target.value }))} className="border rounded px-2 py-1 text-sm" />
                          <select value={editTask.employee_id} onChange={e => setEditTask((x: any) => ({ ...x, employee_id: +e.target.value }))} className="border rounded px-2 py-1 text-sm col-span-2">
                            {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
                          </select>
                          <div className="flex gap-2 col-span-3">
                            <button onClick={() => updateOneTimeTaskMut.mutate({ id: t.id, data: { title: editTask.title, description: editTask.description, due_date: editTask.due_date, employee_id: editTask.employee_id } })} disabled={!editTask.title || updateOneTimeTaskMut.isPending} className="px-3 py-1 bg-green-600 text-white rounded text-xs">Save</button>
                            <button onClick={() => setEditTask(null)} className="px-3 py-1 border rounded text-xs">Cancel</button>
                          </div>
                        </div>
                      </td>
                    ) : (
                      <>
                        <td className="px-4 py-3">
                          <p className="font-medium text-gray-800">{t.title}</p>
                          {t.description && <p className="text-xs text-gray-400 mt-0.5">{t.description}</p>}
                        </td>
                        <td className="px-4 py-3">
                          <p className="font-medium">{t.employee_name}</p>
                          <p className="text-xs text-gray-400">{t.department_name || '—'}</p>
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-600">{t.assigned_by || '—'}</td>
                        <td className="px-4 py-3"><span className={`text-xs px-1.5 py-0.5 rounded ${priorityStyle(t.priority || 'Medium')}`}>{t.priority || 'Medium'}</span></td>
                        <td className="px-4 py-3 text-gray-600">{t.due_date || '—'}</td>
                        <td className="px-4 py-3">
                          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${oneTimeStatusStyle(t.status)}`}>{t.status}</span>
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-500">
                          <p>Start: {fmtDateTime(t.started_at)}</p>
                          <p>End: {fmtDateTime(t.completed_at)}</p>
                          <p className="font-semibold text-[#002B5B] mt-0.5">{fmtDuration(t.duration_minutes)}</p>
                          {manualTimeOpen === t.id ? (
                            <div className="flex gap-1 mt-1">
                              <input value={manualTimeVal} onChange={e => setManualTimeVal(e.target.value)} placeholder="HH:MM" className="border rounded px-1 w-16 text-xs" />
                              <button type="button" className="text-xs text-green-700" onClick={() => manualDurationMut.mutate({ id: t.id, duration: manualTimeVal })}>OK</button>
                              <button type="button" className="text-xs text-gray-500" onClick={() => setManualTimeOpen(null)}>✕</button>
                            </div>
                          ) : (
                            <button type="button" className="text-[10px] text-blue-600 underline mt-1" onClick={() => { setManualTimeOpen(t.id); setManualTimeVal('') }}>Manual time</button>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex flex-wrap gap-1 justify-end">
                            {canMutateRecords && t.status !== 'Approved' && (
                              <button onClick={() => setEditTask({ id: t.id, title: t.title, description: t.description || '', due_date: t.due_date || '', employee_id: t.employee_id })} className="text-xs px-2 py-1 border rounded text-blue-600">✏️ Edit</button>
                            )}
                            {(t.status === 'Pending' || t.status === 'Rejected') && (
                              <button onClick={() => startOneTimeTaskMut.mutate(t.id)} disabled={startOneTimeTaskMut.isPending}
                                className="text-xs px-2 py-1 bg-blue-600 text-white rounded">▶ Start</button>
                            )}
                            {t.status === 'In Progress' && (
                              <button onClick={() => { setCompleteModal({ id: t.id, title: t.title }); setCompleteNotes('') }}
                                className="text-xs px-2 py-1 bg-amber-500 text-white rounded">✓ Mark Done</button>
                            )}
                            {t.status === 'Done' && canAssignTasks && (
                              <>
                                <button onClick={() => { setApprovalModal({ id: t.id, title: t.title, action: 'approve' }); setApprovalNotes('') }}
                                  className="text-xs px-2 py-1 bg-green-600 text-white rounded">Approve</button>
                                <button onClick={() => { setApprovalModal({ id: t.id, title: t.title, action: 'reject' }); setApprovalNotes('') }}
                                  className="text-xs px-2 py-1 bg-red-500 text-white rounded">Reject</button>
                              </>
                            )}
                            {canDeleteHrm && t.status !== 'Approved' && (
                              <button onClick={() => { if (window.confirm('Cancel this task?')) cancelOneTimeTaskMut.mutate(t.id) }}
                                className="text-xs px-2 py-1 border rounded text-red-600">Cancel</button>
                            )}
                          </div>
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
            {filteredTasks.length === 0 && (
              <p className="text-center text-gray-400 py-8 text-sm">No one-time tasks yet.</p>
            )}
          </div>
        </div>
      )}

      {/* ── HOD VIEW ── */}
      {tab === 'hod' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            <select value={hodDept} onChange={e => { setHodDept(e.target.value ? +e.target.value : ''); setHodEmp('') }} className="border rounded-lg px-3 py-1.5 text-sm">
              <option value="">Select Department</option>
              {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
            {hodDept && (
              <select value={hodEmp} onChange={e => setHodEmp(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Employees</option>
                {hodDeptEmployees.map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
              </select>
            )}
            {hodSubTab === 'responsibilities' && (
              <>
                <input type="date" value={fromDate} onChange={e => setFromDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
                <span className="text-gray-400 text-xs">to</span>
                <input type="date" value={toDate} onChange={e => setToDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
              </>
            )}
            {hodSubTab === 'dwr' && (
              <input type="date" value={toDate} onChange={e => setToDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
            )}
          </div>
          <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
            <button onClick={() => setHodSubTab('responsibilities')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium ${hodSubTab === 'responsibilities' ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500'}`}>
              📋 Responsibilities
            </button>
            <button onClick={() => setHodSubTab('tasks')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium ${hodSubTab === 'tasks' ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500'}`}>
              ✅ Tasks {hodPendingTasks.length > 0 && <span className="ml-1 bg-amber-500 text-white px-1.5 rounded-full">{hodPendingTasks.length}</span>}
            </button>
            <button onClick={() => setHodSubTab('dwr')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium ${hodSubTab === 'dwr' ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500'}`}>
              🕒 Daily Work Report
            </button>
          </div>
          {!hodDept && <p className="text-center text-gray-400 py-8 text-sm">Select a department</p>}
          {hodDept && hodSubTab === 'tasks' && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 bg-amber-600 text-white font-semibold">Tasks pending approval</div>
              {(hodPendingTasks as any[]).length === 0 ? (
                <p className="text-center text-gray-400 py-8 text-sm">No tasks awaiting approval.</p>
              ) : (
                <table className="w-full text-sm">
                  <thead className="text-gray-400 text-xs uppercase bg-gray-50">
                    <tr>
                      <th className="text-left px-4 py-2">Task</th>
                      <th className="text-left px-4 py-2">Employee</th>
                      <th className="text-left px-4 py-2">Due</th>
                      <th className="text-left px-4 py-2">Time Taken</th>
                      <th className="px-4 py-2 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(hodPendingTasks as any[]).map((t: any) => (
                      <tr key={t.id} className="border-t">
                        <td className="px-4 py-3">
                          <p className="font-medium">{t.title}</p>
                          {t.completion_notes && <p className="text-xs text-gray-500 mt-0.5">{t.completion_notes}</p>}
                        </td>
                        <td className="px-4 py-3">{t.employee_name}</td>
                        <td className="px-4 py-3">{t.due_date || '—'}</td>
                        <td className="px-4 py-3 text-xs">
                          <p>{fmtDuration(t.duration_minutes)}</p>
                          <p className="text-gray-400">{fmtDateTime(t.completed_at)}</p>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex gap-1 justify-end">
                            <button onClick={() => { setApprovalModal({ id: t.id, title: t.title, action: 'approve' }); setApprovalNotes('') }}
                              className="text-xs px-2 py-1 bg-green-600 text-white rounded">Approve</button>
                            <button onClick={() => { setApprovalModal({ id: t.id, title: t.title, action: 'reject' }); setApprovalNotes('') }}
                              className="text-xs px-2 py-1 bg-red-500 text-white rounded">Reject</button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
          {hodDept && hodSubTab === 'dwr' && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 bg-teal-800 text-white font-semibold">
                Daily Work Report — {toDate}
                {hodEmp ? ` · ${hodDeptEmployees.find((e: any) => e.id === hodEmp)?.name || ''}` : ' · all employees'}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-gray-400 text-xs uppercase bg-gray-50">
                    <tr>
                      <th className="text-left px-3 py-2">Employee</th>
                      <th className="text-left px-3 py-2">Responsibility</th>
                      <th className="text-left px-3 py-2">Status</th>
                      <th className="text-left px-3 py-2">Timer</th>
                      <th className="text-left px-3 py-2">Start</th>
                      <th className="text-left px-3 py-2">End</th>
                      <th className="text-left px-3 py-2">Duration</th>
                      <th className="text-left px-3 py-2">Linked Person</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(dwrData?.rows || []).map((row: any) => (
                      <tr key={`${row.employee_id}-${row.responsibility_id}`} className="border-t">
                        <td className="px-3 py-2">{row.employee_name}</td>
                        <td className="px-3 py-2">
                          <p className="font-medium">{row.title}</p>
                          <p className="text-[10px] text-gray-400">{row.frequency}</p>
                        </td>
                        <td className="px-3 py-2">{row.status}</td>
                        <td className="px-3 py-2"><span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full ${timerBadgeClass(row.timer_status)}`}>{row.timer_status}</span></td>
                        <td className="px-3 py-2 text-xs">{fmtDateTime(row.started_at)}</td>
                        <td className="px-3 py-2 text-xs">{fmtDateTime(row.ended_at)}</td>
                        <td className="px-3 py-2 text-xs font-semibold">{fmtDuration(row.duration_minutes)}</td>
                        <td className="px-3 py-2 text-xs text-indigo-800">{row.linked_person || row.linked_to_employee_name || 'Self-complete'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!(dwrData?.rows || []).length && (
                  <p className="text-center text-gray-400 py-8 text-sm">No DWR rows for this date. Select an employee and date, or ask the employee to start their responsibilities in Employee Check.</p>
                )}
              </div>
            </div>
          )}
          {hodDept && hodSubTab === 'responsibilities' && hodData && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 bg-[#002B5B] text-white flex justify-between">
                <span className="font-semibold">{deptName(hodDept)} — Task Dashboard</span>
                <div className="flex gap-3 text-xs text-blue-200 flex-wrap">
                  <span>○ Pending</span><span>✅ Done</span><span>⚠️ Partial</span><span>❌ Missed</span><span>🔴 Blocked</span><span>🏖 Leave</span><span>— N/A</span>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="text-xs w-full">
                  <thead>
                    <tr className="bg-gray-50 border-b">
                      <th className="text-left px-3 py-2 sticky left-0 bg-gray-50 z-10 min-w-52">Employee · Task</th>
                      <th className="text-left px-3 py-2 min-w-16">Freq</th>
                      {(hodData.dates || []).map((d: string) => (
                        <th key={d} className="px-1 py-2 text-center min-w-10">
                          <div className="font-semibold">{new Date(d).getDate()}</div>
                          <div className="text-gray-400 font-normal">{new Date(d).toLocaleDateString('en-IN', { weekday: 'short' })}</div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(hodData.responsibilities || []).map((r: any) => (
                      <tr key={r.id} className="border-t hover:bg-gray-50">
                        <td className="px-3 py-2 sticky left-0 bg-white z-10">
                          <p className="font-semibold text-[#002B5B]">{r.employee_name}</p>
                          <p className="text-gray-600">{r.title}</p>
                          <p className="text-[10px] text-indigo-700">{r.linked_to_employee_name ? `Approver: ${r.linked_to_employee_name}` : 'Self-complete'}</p>
                        </td>
                        <td className="px-3 py-2 text-gray-400">{r.frequency}</td>
                        {(hodData.dates || []).map((d: string) => {
                          const dayData = r.dates?.[d] || { status: 'Pending', marked: false }
                          const s = dayData.status
                          const locked = !!dayData.marked
                          const hodCanEdit = canEditAssignments && locked && (dayData.editable !== false)
                          return (
                            <td key={d} className="px-1 py-2 text-center">
                              {hodCanEdit ? (
                                <select
                                  value={s}
                                  onChange={e => handleStatusSelect(r.id, d, e.target.value)}
                                  className={`text-[10px] border rounded px-0.5 py-0.5 max-w-[4.5rem] font-bold ${statusBg(s)}`}
                                  title="HOD/Admin can change saved status">
                                  {TASK_LOG_STATUSES.map(st => <option key={st} value={st}>{st}</option>)}
                                </select>
                              ) : locked ? (
                                <span
                                  title={`${s} (locked)${dayData.blocker_name ? ` — Blocked by ${dayData.blocker_name}` : ''}`}
                                  className={`inline-flex w-7 h-7 items-center justify-center rounded-full text-xs font-bold ${statusBg(s)}`}>
                                  {statusLabel(s)}
                                </span>
                              ) : (
                                <select
                                  defaultValue=""
                                  onChange={e => { const val = e.target.value; handleStatusSelect(r.id, d, val); e.target.value = '' }}
                                  className="text-[10px] border rounded px-0.5 py-0.5 max-w-[4.5rem] bg-white"
                                  title="Select status">
                                  <option value="">Set</option>
                                  {TASK_LOG_STATUSES.map(st => <option key={st} value={st}>{st}</option>)}
                                </select>
                              )}
                              {dayData.blocker_name && <p className="text-xs text-purple-600 mt-0.5">{dayData.blocker_name}</p>}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="px-4 py-2 border-t bg-gray-50 text-xs text-gray-500">
                Select a status for each day. Employees cannot change after save; HOD/Admin can update a saved status if needed. Use Leave or N/A for absent employees.
              </div>
            </div>
          )}
        </div>
      )}

      
      {/* ── HIERARCHY ── */}
      {tab === 'hierarchy' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border p-4">
            <h3 className="font-semibold text-gray-800 mb-2">{t(lang, 'hierarchy')}</h3>
            <p className="text-xs text-gray-500 mb-3">Organization departments, HOD assignment, and employee reporting structure</p>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-semibold text-[#002B5B] mb-2">Departments & HOD</h4>
                <ul className="space-y-2 text-sm">
                  {((orgHierarchy?.departments_flat) || depts as any[]).map((d: any) => (
                    <li key={d.id} className="border rounded-lg p-2">
                      <div className="font-medium">{d.name}</div>
                      <div className="text-xs text-gray-500">HOD: {d.hod_name || '—'} · Parent dept: {d.parent_department_id || '—'}</div>
                      {canManageOrg && (
                        <div className="flex gap-2 mt-1">
                          <input defaultValue={d.hod_name || ''} id={`hod-${d.id}`} className="border rounded px-1 py-0.5 text-xs flex-1" placeholder="HOD name" />
                          <button className="text-xs text-blue-600" onClick={() => {
                            const el = document.getElementById(`hod-${d.id}`) as HTMLInputElement
                            api.patch('/hrm/hierarchy/department', { department_id: d.id, hod_name: el?.value }).then(() => qc.invalidateQueries({ queryKey: ['hrm-hierarchy'] }))
                          }}>Save HOD</button>
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-[#002B5B] mb-2">Reporting structure</h4>
                <ul className="space-y-2 text-sm max-h-96 overflow-auto">
                  {((orgHierarchy?.employees_flat) || allEmps as any[]).map((e: any) => (
                    <li key={e.id} className="border rounded-lg p-2 flex flex-wrap gap-2 items-center">
                      <span className="font-medium">{e.name}</span>
                      <span className="text-xs text-gray-400">{e.emp_code} · {e.department_name}</span>
                      {canManageOrg && (
                        <select className="border rounded text-xs ml-auto" defaultValue={e.reports_to_employee_id || ''}
                          onChange={ev => api.patch('/hrm/hierarchy/reporting', { employee_id: e.id, reports_to_employee_id: ev.target.value ? +ev.target.value : null }).then(() => qc.invalidateQueries({ queryKey: ['hrm-hierarchy'] }))}>
                          <option value="">Reports to…</option>
                          {(allEmps as any[]).filter((x: any) => x.id !== e.id).map((m: any) => (
                            <option key={m.id} value={m.id}>{m.name}</option>
                          ))}
                        </select>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}


      {/* ── ISSUES ── */}
      {tab === 'issues' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex gap-2 flex-wrap items-center">
              {!isEmployeeScope && (
                <select value={selDept} onChange={e => setSelDept(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
                  <option value="">All Departments</option>
                  {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
                </select>
              )}
              <select value={issueStatusFilter} onChange={e => setIssueStatusFilter(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Statuses</option>
                {ISSUE_STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
              <input type="date" value={fromDate} onChange={e => setFromDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
              <input type="date" value={toDate} onChange={e => setToDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
              <input
                value={issueQ}
                onChange={e => setIssueQ(e.target.value)}
                placeholder="Search title, employee, ID…"
                className="border rounded-lg px-3 py-1.5 text-sm min-w-[12rem]"
              />
            </div>
            <button onClick={() => setShowIssueForm(true)} className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium">+ Record Issue</button>
          </div>

          {(showIssueForm || editIssue) && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <div className="flex justify-between items-center">
                <h3 className="font-semibold text-gray-700">{editIssue ? `✏️ Edit Issue #${editIssue.id}` : '⚠️ Record Issue / Problem'}</h3>
                <button type="button" onClick={() => { setShowIssueForm(false); setEditIssue(null); stopIssueVoice() }} className="text-gray-400 text-sm">Close</button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="md:col-span-1">
                  <label className="text-xs text-gray-500">Employee * (active users)</label>
                  <input
                    value={issueEmpSearch}
                    onChange={e => setIssueEmpSearch(e.target.value)}
                    placeholder="Search name, email, phone, ID…"
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                  />
                  <select
                    value={(editIssue || issueForm).subject_user_id || (editIssue || issueForm).employee_id || ''}
                    onChange={e => {
                      const v = e.target.value
                      const u = (issueUsers as any[]).find((x: any) => String(x.id) === v)
                        || (issueUsersForFilter as any[]).find((x: any) => String(x.id) === v)
                      if (editIssue) {
                        setEditIssue((f: any) => ({
                          ...f,
                          subject_user_id: v ? +v : null,
                          employee_id: u?.employee_id || f.employee_id,
                          subject_user_name: u?.display_name,
                        }))
                      } else {
                        setIssueForm(f => ({
                          ...f,
                          subject_user_id: v,
                          employee_id: u?.employee_id || '',
                        }))
                      }
                    }}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                  >
                    <option value="">Select employee / user</option>
                    {(issueUsers as any[]).length > 0
                      ? (issueUsers as any[]).map((u: any) => (
                          <option key={u.id} value={u.id}>{u.search_label || u.display_name}</option>
                        ))
                      : (issueUsersForFilter as any[]).map((u: any) => (
                          <option key={u.id} value={u.id}>{u.search_label || u.display_name}</option>
                        ))}
                    {/* Fallback HR employees not linked as users */}
                    {pickerEmps.map((e: any) => (
                      <option key={`e-${e.id}`} value={`emp:${e.id}`}>{e.name} (HR · {e.emp_code || e.id})</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Caused By (active users)</label>
                  <input
                    value={issueCauseSearch}
                    onChange={e => setIssueCauseSearch(e.target.value)}
                    placeholder="Search caused-by…"
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                  />
                  <select
                    value={(editIssue || issueForm).caused_by_user_id || ''}
                    onChange={e => {
                      const v = e.target.value
                      if (editIssue) setEditIssue((f: any) => ({ ...f, caused_by_user_id: v ? +v : null }))
                      else setIssueForm(f => ({ ...f, caused_by_user_id: v }))
                    }}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                  >
                    <option value="">None / same person ok</option>
                    {(issueUsers as any[]).concat(issueUsersForFilter as any[]).filter((u: any, i: number, a: any[]) => a.findIndex(x => x.id === u.id) === i).map((u: any) => (
                      <option key={u.id} value={u.id}>{u.search_label || u.display_name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Recorded By</label>
                  <input
                    value={editIssue ? (editIssue.recorded_by || recordedByName) : recordedByName}
                    readOnly
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1 bg-gray-50 text-gray-600 cursor-not-allowed"
                    title="Always the logged-in user — not editable"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Issue Type</label>
                  <select
                    value={(editIssue || issueForm).issue_type}
                    onChange={e => editIssue ? setEditIssue((f: any) => ({ ...f, issue_type: e.target.value })) : setIssueForm(f => ({ ...f, issue_type: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                  >
                    {ISSUE_TYPES.map(t => <option key={t}>{t}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Severity</label>
                  <select
                    value={(editIssue || issueForm).severity}
                    onChange={e => editIssue ? setEditIssue((f: any) => ({ ...f, severity: e.target.value })) : setIssueForm(f => ({ ...f, severity: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                  >
                    {SEVERITIES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Status</label>
                  <select
                    value={(editIssue || issueForm).status || 'Open'}
                    onChange={e => editIssue ? setEditIssue((f: any) => ({ ...f, status: e.target.value })) : setIssueForm(f => ({ ...f, status: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                    disabled={!canChangeIssueStatus && !!editIssue}
                  >
                    {ISSUE_STATUSES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
                <div className="md:col-span-2">
                  <div className="flex items-center justify-between gap-2">
                    <label className="text-xs text-gray-500">Issue Title *</label>
                    <button type="button" onClick={() => startIssueVoice('title')}
                      className={`text-xs px-2 py-0.5 rounded border ${issueListening && issueVoiceTarget === 'title' ? 'bg-red-100 border-red-300 text-red-700' : 'bg-white'}`}>
                      🎤 {issueListening && issueVoiceTarget === 'title' ? 'Pause' : 'Voice'}
                    </button>
                  </div>
                  <input
                    value={(editIssue || issueForm).title || ''}
                    onChange={e => editIssue ? setEditIssue((f: any) => ({ ...f, title: e.target.value })) : setIssueForm(f => ({ ...f, title: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                  />
                </div>
                <div className="md:col-span-3">
                  <div className="flex items-center justify-between gap-2">
                    <label className="text-xs text-gray-500">Description</label>
                    <button type="button" onClick={() => startIssueVoice('description')}
                      className={`text-xs px-2 py-0.5 rounded border ${issueListening && issueVoiceTarget === 'description' ? 'bg-red-100 border-red-300 text-red-700' : 'bg-white'}`}>
                      🎤 {issueListening && issueVoiceTarget === 'description' ? 'Pause' : 'Dictate'}
                    </button>
                  </div>
                  <textarea
                    value={(editIssue || issueForm).description || ''}
                    onChange={e => editIssue ? setEditIssue((f: any) => ({ ...f, description: e.target.value })) : setIssueForm(f => ({ ...f, description: e.target.value }))}
                    rows={3}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                  />
                  {issueVoiceStatus && <p className="text-[11px] text-blue-700 mt-1">{issueVoiceStatus}</p>}
                </div>
              </div>
              {editIssue && canEditIssues && (
                <div className="grid md:grid-cols-2 gap-2 text-xs border-t pt-3">
                  <div className="flex gap-1">
                    <input value={issueComment} onChange={e => setIssueComment(e.target.value)} placeholder="Add comment…" className="flex-1 border rounded px-2 py-1.5" />
                    <button type="button" disabled={!issueComment.trim()} onClick={() => commentIssueMut.mutate({ id: editIssue.id, text: issueComment })}
                      className="px-2 py-1 bg-gray-800 text-white rounded disabled:opacity-50">Comment</button>
                  </div>
                  <div className="flex gap-1">
                    <input value={issueAttachmentName} onChange={e => setIssueAttachmentName(e.target.value)} placeholder="Attachment file name…" className="flex-1 border rounded px-2 py-1.5" />
                    <button type="button" disabled={!issueAttachmentName.trim()} onClick={() => attachIssueMut.mutate({ id: editIssue.id, file_name: issueAttachmentName })}
                      className="px-2 py-1 border rounded disabled:opacity-50">Attach</button>
                  </div>
                </div>
              )}
              <div className="flex gap-2 flex-wrap">
                {!editIssue ? (
                  <button
                    onClick={() => {
                      const f = issueForm
                      const payload: any = {
                        title: f.title,
                        description: f.description,
                        issue_type: f.issue_type,
                        severity: f.severity,
                        status: f.status || 'Open',
                      }
                      if (String(f.subject_user_id).startsWith('emp:')) {
                        payload.employee_id = +String(f.subject_user_id).replace('emp:', '')
                      } else if (f.subject_user_id) {
                        payload.subject_user_id = +f.subject_user_id
                        if (f.employee_id) payload.employee_id = +f.employee_id
                      } else if (f.employee_id) {
                        payload.employee_id = +f.employee_id
                      }
                      if (f.caused_by_user_id) payload.caused_by_user_id = +f.caused_by_user_id
                      createIssueMut.mutate(payload)
                    }}
                    disabled={
                      !(issueForm.subject_user_id || issueForm.employee_id) || !issueForm.title || createIssueMut.isPending
                    }
                    className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm disabled:opacity-50"
                  >
                    Record Issue
                  </button>
                ) : canEditIssues ? (
                  <button
                    onClick={() => {
                      const f = editIssue
                      const data: any = {
                        title: f.title,
                        description: f.description,
                        issue_type: f.issue_type,
                        severity: f.severity,
                        status: f.status,
                      }
                      if (f.subject_user_id) data.subject_user_id = +f.subject_user_id
                      if (f.employee_id) data.employee_id = +f.employee_id
                      if (f.caused_by_user_id) data.caused_by_user_id = +f.caused_by_user_id
                      updateIssueMut.mutate({ id: f.id, data })
                    }}
                    className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm"
                  >
                    Save Changes
                  </button>
                ) : null}
                <button onClick={() => { setShowIssueForm(false); setEditIssue(null); stopIssueVoice() }} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
              </div>
            </div>
          )}

          <div className="bg-white rounded-xl border overflow-x-auto">
            <table className="w-full text-sm min-w-[720px]">
              <thead className="bg-gray-50 text-xs text-gray-500">
                <tr>
                  <th className="text-left px-3 py-2">Employee</th>
                  <th className="text-left px-3 py-2">Issue Title</th>
                  <th className="text-left px-3 py-2">Caused By</th>
                  <th className="text-left px-3 py-2">Recorded By</th>
                  <th className="text-left px-3 py-2">Status</th>
                  <th className="text-left px-3 py-2">Created</th>
                  <th className="text-left px-3 py-2">Updated</th>
                  <th className="text-left px-3 py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {(issues as any[]).map((issue: any) => {
                  const empName = issue.display_employee || issue.employee_name || '—'
                  const causeName = issue.display_caused_by || issue.caused_by_name || '—'
                  const showRec = issue.show_recorded_by !== false && issue.recorded_by
                    && (issue.recorded_by || '').toLowerCase() !== (empName || '').toLowerCase()
                  const st = issue.status === 'Resolved' ? 'Resolve' : issue.status
                  return (
                    <tr key={issue.id} className="border-t hover:bg-gray-50">
                      <td className="px-3 py-2 font-medium text-gray-800">
                        {empName}
                        <div className="text-[10px] text-gray-400">{issue.department_name}</div>
                      </td>
                      <td className="px-3 py-2">
                        <div className="font-medium text-gray-700">{issue.title}</div>
                        {issue.description && <div className="text-[11px] text-gray-400 line-clamp-1">{issue.description}</div>}
                      </td>
                      <td className="px-3 py-2 text-purple-700 text-xs">{causeName}</td>
                      <td className="px-3 py-2 text-xs text-gray-600">{showRec ? issue.recorded_by : '—'}</td>
                      <td className="px-3 py-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${issueStatusStyle(st)}`}>{st}</span>
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-500 whitespace-nowrap">{fmtDate(issue.created_at || issue.issue_date)}</td>
                      <td className="px-3 py-2 text-xs text-gray-500 whitespace-nowrap">{fmtDateTime(issue.updated_at || issue.created_at || '')}</td>
                      <td className="px-3 py-2">
                        <div className="flex flex-wrap gap-1">
                          {canEditIssues && (
                            <button type="button" onClick={() => { setEditIssue({ ...issue }); setShowIssueForm(false) }}
                              className="text-xs px-2 py-0.5 border rounded hover:bg-gray-100">Edit</button>
                          )}
                          <button type="button" onClick={() => openIssueHistory(issue.id)}
                            className="text-xs px-2 py-0.5 border rounded hover:bg-gray-100">History</button>
                          {canChangeIssueStatus && st === 'Open' && (
                            <>
                              <button type="button" onClick={() => statusIssueMut.mutate({ id: issue.id, status: 'Hold' })}
                                className="text-xs px-2 py-0.5 rounded bg-orange-100 text-orange-800">Hold</button>
                              <button type="button" onClick={() => {
                                const res = prompt('Resolution notes (optional):') ?? ''
                                resolveIssueMut.mutate({ id: issue.id, res })
                              }} className="text-xs px-2 py-0.5 rounded bg-green-600 text-white">Resolve</button>
                            </>
                          )}
                          {canChangeIssueStatus && st === 'Hold' && (
                            <button type="button" onClick={() => statusIssueMut.mutate({ id: issue.id, status: 'Open' })}
                              className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-800">Reopen</button>
                          )}
                          {canChangeIssueStatus && st !== 'Cancel' && st !== 'Resolve' && (
                            <button type="button" onClick={() => {
                              if (confirm('Cancel this issue?')) statusIssueMut.mutate({ id: issue.id, status: 'Cancel' })
                            }} className="text-xs px-2 py-0.5 rounded bg-red-50 text-red-700">Cancel</button>
                          )}
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            {(issues as any[]).length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No issues found.</p>}
          </div>

          {showHistoryId && (
            <div className="bg-white rounded-xl border p-4">
              <div className="flex justify-between items-center mb-2">
                <h4 className="text-sm font-semibold">Audit trail · Issue #{showHistoryId}</h4>
                <button type="button" className="text-xs text-gray-500" onClick={() => setShowHistoryId(null)}>Close</button>
              </div>
              <ul className="space-y-1 max-h-48 overflow-auto text-xs">
                {issueHistory.map((h: any) => (
                  <li key={h.id} className="border-b border-gray-50 py-1">
                    <span className="text-gray-400">{h.created_at}</span>{' '}
                    <b>{h.action}</b>
                    {h.field_name ? ` · ${h.field_name}` : ''}
                    {h.previous_value || h.new_value ? (
                      <span className="text-gray-600">: {h.previous_value || '∅'} → {h.new_value || '∅'}</span>
                    ) : null}
                    <span className="text-gray-400"> · {h.user_name || 'system'}</span>
                  </li>
                ))}
                {issueHistory.length === 0 && <li className="text-gray-400">No history yet.</li>}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* ── APPRAISAL ── */}
      {tab === 'appraisal' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            {!isEmployeeScope ? (
              <select value={appraisalEmp} onChange={e => setAppraisalEmp(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
                <option value="">Select Employee</option>
                {pickerEmps.map((e: any) => <option key={e.id} value={e.id}>{e.name} — {e.department_name || '—'}</option>)}
              </select>
            ) : (
              <span className="text-sm font-medium text-gray-700">Your appraisal</span>
            )}
            <input type="date" value={appraisalFrom} onChange={e => setAppraisalFrom(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
            <span className="text-gray-400 text-xs">to</span>
            <input type="date" value={appraisalTo} onChange={e => setAppraisalTo(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
          </div>
          {!appraisalEmp && <p className="text-center text-gray-400 py-8 text-sm">Select an employee to view appraisal</p>}
          {appraisalData && (
            <div className="space-y-4">
              {/* Header */}
              <div className="bg-[#002B5B] text-white rounded-xl p-4">
                <h3 className="font-bold text-lg">{appraisalData.employee?.name}</h3>
                <p className="text-blue-200 text-sm">{appraisalData.employee?.department_name} · {appraisalData.employee?.designation}</p>
                <p className="text-blue-300 text-xs mt-1">{appraisalData.period?.from} to {appraisalData.period?.to}</p>
              </div>
              {/* Responsibility summary */}
              <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
                {[
                  ['Resp. Total', appraisalData.task_summary?.total, 'text-gray-700'],
                  ['Done ✅', appraisalData.task_summary?.done, 'text-green-600'],
                  ['Partial ⚠️', appraisalData.task_summary?.partial, 'text-amber-600'],
                  ['Missed ❌', appraisalData.task_summary?.missed, 'text-red-600'],
                  ['Blocked 🔴', appraisalData.task_summary?.blocked, 'text-purple-600'],
                  ['Resp. Score', `${appraisalData.task_summary?.responsibility_performance_pct ?? appraisalData.task_summary?.performance_pct}%`, 'text-blue-600'],
                ].map(([l, v, c]) => (
                  <div key={l as string} className="bg-white rounded-xl border p-3 text-center">
                    <p className={`text-xl font-bold ${c}`}>{v}</p>
                    <p className="text-xs text-gray-400 mt-1">{l}</p>
                  </div>
                ))}
              </div>
              {appraisalData.one_time_summary?.total > 0 && (
                <div className="bg-white rounded-xl border p-4 space-y-3">
                  <h4 className="font-semibold text-[#002B5B]">✅ One-Time Tasks (impacts score)</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center text-sm">
                    <div><p className="text-xl font-bold text-green-600">{appraisalData.one_time_summary.approved_on_time}</p><p className="text-xs text-gray-400">On time ✅</p></div>
                    <div><p className="text-xl font-bold text-amber-600">{appraisalData.one_time_summary.awaiting_approval}</p><p className="text-xs text-gray-400">Awaiting HOD</p></div>
                    <div><p className="text-xl font-bold text-red-600">{appraisalData.one_time_summary.overdue + appraisalData.one_time_summary.pending}</p><p className="text-xs text-gray-400">Pending / Overdue</p></div>
                    <div><p className="text-xl font-bold text-red-700">{appraisalData.one_time_summary.rejected}</p><p className="text-xs text-gray-400">Rejected</p></div>
                  </div>
                  <p className="text-sm text-center">
                    Task score: <b className={appraisalData.one_time_summary.performance_pct >= 80 ? 'text-green-600' : appraisalData.one_time_summary.performance_pct >= 50 ? 'text-amber-600' : 'text-red-600'}>
                      {appraisalData.one_time_summary.performance_pct}%
                    </b>
                    {' · '}Combined: <b className="text-[#002B5B]">{appraisalData.task_summary?.performance_pct}%</b>
                  </p>
                </div>
              )}
              {/* Issues */}
              {appraisalData.issues?.length > 0 && (
                <div className="bg-white rounded-xl border p-4">
                  <h4 className="font-semibold text-red-600 mb-2">⚠️ Issues Recorded ({appraisalData.issues.length})</h4>
                  <table className="w-full text-xs">
                    <thead><tr className="text-gray-400 border-b"><th className="text-left py-1">Date</th><th className="text-left py-1">Type</th><th className="text-left py-1">Severity</th><th className="text-left py-1">Title</th><th className="text-left py-1">Status</th></tr></thead>
                    <tbody>{appraisalData.issues.map((i: any) => (
                      <tr key={i.id} className="border-t">
                        <td className="py-1.5">{i.issue_date}</td>
                        <td className="py-1.5">{i.issue_type}</td>
                        <td className={`py-1.5 font-medium ${i.severity === 'Major' ? 'text-red-600' : i.severity === 'Moderate' ? 'text-amber-600' : 'text-yellow-600'}`}>{i.severity}</td>
                        <td className="py-1.5">{i.title}</td>
                        <td className="py-1.5"><span className={`px-1.5 py-0.5 rounded text-xs ${i.status === 'Resolved' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>{i.status}</span></td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              )}
              {/* Blockers caused */}
              {appraisalData.blockers_caused?.length > 0 && (
                <div className="bg-white rounded-xl border p-4">
                  <h4 className="font-semibold text-purple-600 mb-2">🔴 Blockers caused — blocked others&apos; work ({appraisalData.blockers_caused.length})</h4>
                  <table className="w-full text-xs">
                    <thead><tr className="text-gray-400 border-b"><th className="text-left py-1">Date</th><th className="text-left py-1">Affected Employee</th><th className="text-left py-1">Task</th><th className="text-left py-1">Reason</th></tr></thead>
                    <tbody>{appraisalData.blockers_caused.map((b: any, i: number) => (
                      <tr key={i} className="border-t">
                        <td className="py-1.5">{b.log_date}</td>
                        <td className="py-1.5 font-medium">{b.affected_employee}</td>
                        <td className="py-1.5">{b.task_title}</td>
                        <td className="py-1.5 text-gray-500">{b.blocker_reason || '—'}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── PERFORMANCE ── */}
      {tab === 'performance' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border overflow-hidden">
            <div className="px-4 py-3 bg-[#002B5B] text-white flex justify-between items-center flex-wrap gap-2">
              <h3 className="font-semibold">{t(lang, 'taskReport')}</h3>
              <div className="flex gap-2 text-xs">
                <input type="date" value={fromDate} onChange={e => setFromDate(e.target.value)} className="rounded text-gray-800 px-1" />
                <input type="date" value={toDate} onChange={e => setToDate(e.target.value)} className="rounded text-gray-800 px-1" />
                <select value={taskPriorityFilter} onChange={e => setTaskPriorityFilter(e.target.value)} className="rounded text-gray-800 px-1">
                  <option value="">Priority</option>
                  {PRIORITIES.map(p => <option key={p}>{p}</option>)}
                </select>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 text-gray-500">
                  <tr>
                    <th className="text-left px-3 py-2">Task</th>
                    <th className="text-left px-3 py-2">{t(lang, 'assignedTo')}</th>
                    <th className="text-left px-3 py-2">{t(lang, 'assignedBy')}</th>
                    <th className="text-left px-3 py-2">{t(lang, 'departments')}</th>
                    <th className="text-left px-3 py-2">{t(lang, 'dueDate')}</th>
                    <th className="text-left px-3 py-2">{t(lang, 'status')}</th>
                    <th className="text-right px-3 py-2">{t(lang, 'completion')}</th>
                    <th className="text-left px-3 py-2">{t(lang, 'priority')}</th>
                    <th className="text-left px-3 py-2">{t(lang, 'frequency')}</th>
                    <th className="text-left px-3 py-2">{t(lang, 'mandatory')}</th>
                  </tr>
                </thead>
                <tbody>
                  {(taskReport as any[]).map((row: any, i: number) => (
                    <tr key={i} className="border-t">
                      <td className="px-3 py-1.5 font-medium">{row.task}</td>
                      <td className="px-3 py-1.5">{row.assigned_to}</td>
                      <td className="px-3 py-1.5">{row.assigned_by || '—'}</td>
                      <td className="px-3 py-1.5">{row.department || '—'}</td>
                      <td className="px-3 py-1.5">{row.due_date || '—'}</td>
                      <td className="px-3 py-1.5">{row.status}</td>
                      <td className="px-3 py-1.5 text-right">{row.completion_pct}%</td>
                      <td className="px-3 py-1.5"><span className={`px-1.5 py-0.5 rounded ${priorityStyle(row.priority)}`}>{row.priority}</span></td>
                      <td className="px-3 py-1.5">{row.frequency}</td>
                      <td className="px-3 py-1.5">{row.mandatory ? t(lang, 'yes') : t(lang, 'no')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {(taskReport as any[]).length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No task report rows for this period.</p>}
            </div>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <select value={selDept} onChange={e => setSelDept(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
              <option value="">All Departments</option>
              {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
            <select value={selEmp} onChange={e => setSelEmp(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
              <option value="">All Employees</option>
              {pickerEmps.map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
            </select>
            <input type="date" value={fromDate} onChange={e => setFromDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
            <input type="date" value={toDate} onChange={e => setToDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
          </div>
          <div className="space-y-3">
            {(perfData as any[]).map((p: any, i: number) => (
              <div key={i} className="bg-white rounded-xl border p-4">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <p className="font-semibold text-gray-800">{p.employee_name}</p>
                    <p className="text-xs text-gray-500">{p.department_name}</p>
                    <div className="flex gap-3 text-xs mt-1 flex-wrap">
                      <span className="text-green-600">✅ Done: {p.done_tasks}</span>
                      <span className="text-red-500">❌ Missed: {p.missed_tasks}</span>
                      <span className="text-purple-600">🔴 Blocked: {p.blocked_tasks}</span>
                      {p.one_time_summary?.total > 0 && (
                        <>
                          <span className="text-green-700">✅ Tasks on time: {p.one_time_summary.approved_on_time}</span>
                          <span className="text-red-600">⏳ Overdue/Pending: {(p.one_time_summary.overdue || 0) + (p.one_time_summary.pending || 0)}</span>
                          {p.one_time_summary.rejected > 0 && <span className="text-red-700">↩ Rejected: {p.one_time_summary.rejected}</span>}
                        </>
                      )}
                      {p.issues_total > 0 && <span className="text-amber-600">⚠️ Issues: {p.issues_total} ({p.issues_major} major)</span>}
                      {p.blockers_caused > 0 && <span className="text-purple-700">🔴 Caused blocks: {p.blockers_caused}</span>}
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`text-2xl font-bold ${p.performance_pct >= 80 ? 'text-green-600' : p.performance_pct >= 50 ? 'text-amber-600' : 'text-red-600'}`}>{p.performance_pct}%</p>
                    <button onClick={() => { setAppraisalEmp(p.employee_id); setTab('appraisal') }} className="text-xs text-blue-600 underline">📁 Full Appraisal</button>
                  </div>
                </div>
                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${p.performance_pct >= 80 ? 'bg-green-500' : p.performance_pct >= 50 ? 'bg-amber-400' : 'bg-red-500'}`} style={{ width: `${p.performance_pct}%` }} />
                </div>
              </div>
            ))}
            {(perfData as any[]).length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No data yet. Mark tasks in HOD view first.</p>}
          </div>
        </div>
      )}

      {/* ── COMPLETE TASK MODAL ── */}
      {completeModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <h3 className="font-semibold text-amber-700">✓ Mark task complete</h3>
            <p className="text-sm text-gray-600">{completeModal.title}</p>
            <div>
              <label className="text-xs text-gray-500">Completion notes (optional)</label>
              <textarea value={completeNotes} onChange={e => setCompleteNotes(e.target.value)} rows={3}
                className="w-full border rounded px-2 py-1.5 text-sm mt-1" placeholder="What was done?" />
            </div>
            <div className="flex gap-2">
              <button onClick={() => completeOneTimeTaskMut.mutate({ id: completeModal.id, notes: completeNotes })}
                disabled={completeOneTimeTaskMut.isPending}
                className="flex-1 py-2 bg-amber-500 text-white rounded-lg text-sm disabled:opacity-50">
                {completeOneTimeTaskMut.isPending ? 'Saving…' : 'Submit for approval'}
              </button>
              <button onClick={() => setCompleteModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* ── APPROVAL MODAL ── */}
      {approvalModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <h3 className={`font-semibold ${approvalModal.action === 'approve' ? 'text-green-700' : 'text-red-700'}`}>
              {approvalModal.action === 'approve' ? '✅ Approve task' : '↩ Reject task'}
            </h3>
            <p className="text-sm text-gray-600">{approvalModal.title}</p>
            <div>
              <label className="text-xs text-gray-500">Notes (optional)</label>
              <textarea value={approvalNotes} onChange={e => setApprovalNotes(e.target.value)} rows={3}
                className="w-full border rounded px-2 py-1.5 text-sm mt-1" />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  if (approvalModal.action === 'approve') {
                    approveOneTimeTaskMut.mutate({ id: approvalModal.id, notes: approvalNotes })
                  } else {
                    rejectOneTimeTaskMut.mutate({ id: approvalModal.id, notes: approvalNotes })
                  }
                }}
                disabled={approveOneTimeTaskMut.isPending || rejectOneTimeTaskMut.isPending}
                className={`flex-1 py-2 text-white rounded-lg text-sm disabled:opacity-50 ${approvalModal.action === 'approve' ? 'bg-green-600' : 'bg-red-500'}`}>
                {approvalModal.action === 'approve' ? 'Approve & close' : 'Reject — send back'}
              </button>
              <button onClick={() => setApprovalModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* ── BLOCKED MODAL ── */}
      {blockedModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <h3 className="font-semibold text-purple-700">🔴 Task blocked — who is responsible?</h3>
            <p className="text-xs text-gray-500 bg-purple-50 px-3 py-2 rounded-lg">
              This task is blocked. An issue will be added automatically to the blocker employee&apos;s record.
            </p>
            <div className="space-y-3">
              <div><label className="text-xs text-gray-500">Blocker Employee *</label>
                <select value={blockedForm.blocker_employee_id} onChange={e => setBlockedForm(f => ({ ...f, blocker_employee_id: e.target.value }))}
                  className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                  <option value="">Select employee</option>
                  {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name} ({e.department_name || '—'})</option>)}
                </select>
              </div>
              <div><label className="text-xs text-gray-500">Reason *</label>
                <input value={blockedForm.blocker_reason} onChange={e => setBlockedForm(f => ({ ...f, blocker_reason: e.target.value }))}
                  placeholder="e.g. Cutting data not provided"
                  className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
              <div><label className="text-xs text-gray-500">Marked By</label>
                <input value={blockedForm.marked_by} onChange={e => setBlockedForm(f => ({ ...f, marked_by: e.target.value }))}
                  className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => markTaskMut.mutate({
                responsibility_id: blockedModal.respId,
                log_date: blockedModal.date,
                status: 'Blocked',
                marked_by: blockedForm.marked_by,
                blocker_employee_id: blockedForm.blocker_employee_id ? +blockedForm.blocker_employee_id : null,
                blocker_reason: blockedForm.blocker_reason,
              })}
                disabled={markTaskMut.isPending || !blockedForm.blocker_employee_id || !blockedForm.blocker_reason}
                className="flex-1 py-2 bg-purple-600 text-white rounded-lg text-sm disabled:opacity-50">
                {markTaskMut.isPending ? 'Saving…' : '🔴 Mark Blocked + Auto Issue'}
              </button>
              <button onClick={() => setBlockedModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
