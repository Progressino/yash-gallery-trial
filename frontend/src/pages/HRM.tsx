import { useState, useEffect, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import { useAuth } from '../store/auth'

type Tab = 'dashboard' | 'employees' | 'responsibilities' | 'tasks' | 'hod' | 'issues' | 'appraisal' | 'performance'

const FREQUENCIES = ['Daily', 'Weekly', 'Monthly']
const ONE_TIME_STATUSES = ['Pending', 'In Progress', 'Done', 'Approved', 'Rejected'] as const
type ItemType = 'responsibility' | 'task'
const CATEGORIES = ['General', 'Quality', 'Production', 'Accounts', 'Purchase', 'Sales', 'Store', 'HR', 'Other']
const ISSUE_TYPES = ['General', 'Discipline', 'Quality', 'Attendance', 'Behaviour', 'Task Failure', 'Dependency Missed']
const SEVERITIES = ['Minor', 'Moderate', 'Major']

const today = () => new Date().toISOString().split('T')[0]
const fmt7Days = () => { const d = new Date(); d.setDate(d.getDate() - 6); return d.toISOString().split('T')[0] }
const fmtMonth = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-01` }

const statusLabel = (s: string) => {
  if (s === 'Done') return '✅'
  if (s === 'Partial') return '⚠️'
  if (s === 'Missed') return '❌'
  if (s === 'Blocked') return '🔴'
  return '○'
}

const statusNext = (s: string) => {
  if (s === 'Pending') return 'Done'
  if (s === 'Done') return 'Partial'
  if (s === 'Partial') return 'Missed'
  if (s === 'Missed') return 'Blocked'
  return 'Pending'
}

const statusBg = (s: string) => {
  if (s === 'Done') return 'bg-green-500 text-white'
  if (s === 'Partial') return 'bg-yellow-400 text-white'
  if (s === 'Missed') return 'bg-red-500 text-white'
  if (s === 'Blocked') return 'bg-purple-600 text-white'
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

const fmtDateTime = (iso: string) => {
  if (!iso) return '—'
  const d = new Date(iso.replace(' ', 'T'))
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
}

export default function HRM() {
  const qc = useQueryClient()
  const authUser = useAuth(s => s.user)
  const { data: scopeApi } = useQuery({
    queryKey: ['hrm-scope'],
    queryFn: () => api.get('/hrm/scope').then(r => r.data),
  })
  const scope = scopeApi || authUser?.hrm_scope
  const canManageOrg = scope?.can_manage_org ?? (authUser?.role === 'Super Admin' || authUser?.role === 'Admin' || authUser?.role === 'Sir')
  const scopeLevel = scope?.level || 'all'
  const isEmployeeScope = scopeLevel === 'self'
  const canAssignTasks = !isEmployeeScope
  const canEditAssignments = scope?.can_edit_assignments ?? (canManageOrg || scope?.role === 'HOD')

  const [tab, setTab] = useState<Tab>('dashboard')
  const [selDept, setSelDept] = useState<number | ''>('')
  const [selEmp, setSelEmp] = useState<number | ''>('')
  const [hodDept, setHodDept] = useState<number | ''>('')
  const [appraisalEmp, setAppraisalEmp] = useState<number | ''>('')
  const [fromDate, setFromDate] = useState(fmt7Days())
  const [toDate, setToDate] = useState(today())
  const [appraisalFrom, setAppraisalFrom] = useState(fmtMonth())
  const [appraisalTo, setAppraisalTo] = useState(today())

  const [showDeptForm, setShowDeptForm] = useState(false)
  const [showEmpForm, setShowEmpForm] = useState(false)
  const [showRespForm, setShowRespForm] = useState(false)
  const [showTaskForm, setShowTaskForm] = useState(false)
  const [showIssueForm, setShowIssueForm] = useState(false)
  const [taskStatusFilter, setTaskStatusFilter] = useState('')
  const [completeModal, setCompleteModal] = useState<{ id: number; title: string } | null>(null)
  const [completeNotes, setCompleteNotes] = useState('')
  const [approvalModal, setApprovalModal] = useState<{ id: number; title: string; action: 'approve' | 'reject' } | null>(null)
  const [approvalNotes, setApprovalNotes] = useState('')
  const [hodSubTab, setHodSubTab] = useState<'responsibilities' | 'tasks'>('responsibilities')
  const [editDept, setEditDept] = useState<any>(null)
  const [editEmp, setEditEmp] = useState<any>(null)
  const [editResp, setEditResp] = useState<any>(null)
  const [editTask, setEditTask] = useState<any>(null)

  // Blocked modal
  const [blockedModal, setBlockedModal] = useState<{ respId: number; date: string } | null>(null)
  const [blockedForm, setBlockedForm] = useState({ blocker_employee_id: '' as any, blocker_reason: '', marked_by: '' })

  const [deptForm, setDeptForm] = useState({ name: '', description: '', hod_name: '' })
  const [empForm, setEmpForm] = useState({ name: '', department_id: '' as any, designation: '', phone: '', email: '', join_date: '' })
  const [respForm, setRespForm] = useState({ item_type: 'responsibility' as ItemType, employee_id: '' as any, title: '', description: '', frequency: 'Daily', category: 'General', added_by: '', due_date: '' })
  const [taskForm, setTaskForm] = useState({ employee_id: '' as any, title: '', description: '', due_date: '', assigned_by: '' })
  const [issueForm, setIssueForm] = useState({ employee_id: '' as any, issue_type: 'General', severity: 'Minor', title: '', description: '', recorded_by: '', caused_by_employee_id: '' as any })

  // Quick resp + voice
  const [quickResp, setQuickResp] = useState({ item_type: 'responsibility' as ItemType, employee_id: '' as any, department_id: '' as any, title: '', frequency: 'Daily', category: 'General', added_by: '', due_date: '' })
  const [showQuickResp, setShowQuickResp] = useState(false)
  const [voiceText, setVoiceText] = useState('')
  const [isListening, setIsListening] = useState(false)
  const [aiParsing, setAiParsing] = useState(false)
  const [aiParsed, setAiParsed] = useState<any>(null)

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
    else if (lowerText.includes('monthly')) frequency = 'Monthly'
    const taskHints = ['by friday', 'by monday', 'by tuesday', 'by wednesday', 'by thursday', 'by saturday', 'by sunday', 'one time', 'one-time', 'audit', 'complete the', 'finish the', 'before ']
    let itemType: ItemType = taskHints.some(h => lowerText.includes(h)) ? 'task' : 'responsibility'
    let dueDate = ''
    if (itemType === 'task') {
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
    title = title.replace(/\b(from now on|daily|weekly|monthly|will|shall|must|the|and|or)\b/gi, '').replace(/\s+/g, ' ').trim()
    if (!title) title = text
    const parsed = { item_type: itemType, employee_id: matchedEmp?.id || null, employee_name: matchedEmp?.name || '', department_id: matchedEmp?.department_id || null, title, frequency, category: 'General', due_date: dueDate }
    setAiParsed(parsed)
    setQuickResp({ item_type: itemType, employee_id: parsed.employee_id || '', department_id: parsed.department_id || '', title: parsed.title, frequency: parsed.frequency, category: 'General', added_by: '', due_date: dueDate })
    setShowQuickResp(true)
    setAiParsing(false)
  }

  const startListening = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) { alert('Speech recognition requires Chrome or Edge.'); return }
    const recognition = new SR()
    recognition.lang = 'en-IN'
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
    queryKey: ['hrm-hod', hodDept, fromDate, toDate],
    queryFn: () => api.get(`/hrm/hod-dashboard/${hodDept}?from_date=${fromDate}&to_date=${toDate}`).then(r => r.data),
    enabled: !!hodDept,
  })
  const { data: issues = [] } = useQuery({
    queryKey: ['hrm-issues', selDept, selEmp, fromDate, toDate],
    queryFn: () => api.get(`/hrm/issues?${selDept ? `department_id=${selDept}&` : ''}${selEmp ? `employee_id=${selEmp}&` : ''}from_date=${fromDate}&to_date=${toDate}`).then(r => r.data),
    enabled: tab === 'issues',
  })
  const { data: appraisalData } = useQuery({
    queryKey: ['hrm-appraisal', appraisalEmp, appraisalFrom, appraisalTo],
    queryFn: () => api.get(`/hrm/appraisal/${appraisalEmp}?from_date=${appraisalFrom}&to_date=${appraisalTo}`).then(r => r.data),
    enabled: !!appraisalEmp,
  })
  const { data: perfData = [] } = useQuery({
    queryKey: ['hrm-perf', selDept, fromDate, toDate],
    queryFn: () => api.get(`/hrm/performance?from_date=${fromDate}&to_date=${toDate}${selDept ? `&department_id=${selDept}` : ''}`).then(r => r.data),
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
  const createRespMut = useMutation({ mutationFn: (b: object) => api.post('/hrm/responsibilities', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-resps'] }); qc.invalidateQueries({ queryKey: ['hrm-hod'] }); setShowRespForm(false); setShowQuickResp(false); setAiParsed(null); setVoiceText('') } })
  const updateRespMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/hrm/responsibilities/${id}`, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-resps'] }); qc.invalidateQueries({ queryKey: ['hrm-hod'] }); setEditResp(null) },
  })
  const deleteRespMut = useMutation({ mutationFn: (id: number) => api.delete(`/hrm/responsibilities/${id}`), onSuccess: () => qc.invalidateQueries({ queryKey: ['hrm-resps'] }) })
  const markTaskMut = useMutation({
    mutationFn: (b: object) => api.post('/hrm/tasks/mark', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-hod'] }); qc.invalidateQueries({ queryKey: ['hrm-perf'] }); setBlockedModal(null) }
  })
  const createIssueMut = useMutation({ mutationFn: (b: object) => api.post('/hrm/issues', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['hrm-issues'] }); setShowIssueForm(false) } })
  const resolveIssueMut = useMutation({ mutationFn: ({ id, res }: { id: number; res: string }) => api.patch(`/hrm/issues/${id}/resolve`, { resolution: res }), onSuccess: () => qc.invalidateQueries({ queryKey: ['hrm-issues'] }) })
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
      setTaskForm({ employee_id: '', title: '', description: '', due_date: '', assigned_by: '' })
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
  const submitAssign = (form: typeof quickResp) => {
    if (!form.employee_id || !form.title) return
    if (form.item_type === 'task') {
      createOneTimeTaskMut.mutate({
        employee_id: +form.employee_id,
        title: form.title,
        description: '',
        due_date: form.due_date || '',
        assigned_by: form.added_by || '',
      })
    } else {
      createRespMut.mutate({
        employee_id: +form.employee_id,
        department_id: form.department_id ? +form.department_id : null,
        title: form.title,
        frequency: form.frequency,
        category: form.category,
        added_by: form.added_by || '',
      })
    }
  }

  const deptName = (id: any) => (depts as any[]).find(d => d.id === id)?.name || '—'

  const ALL_TABS: [Tab, string][] = [
    ['dashboard', '📊 Dashboard'],
    ['employees', '👥 Employees'],
    ['responsibilities', '📋 Responsibilities'],
    ['tasks', '✅ Tasks'],
    ['hod', '🏢 HOD View'],
    ['issues', '⚠️ Issues'],
    ['appraisal', '📁 Appraisal'],
    ['performance', '📈 Performance'],
  ]

  const TABS = useMemo(() => {
    if (scopeLevel === 'self') {
      return ALL_TABS.filter(([k]) => ['dashboard', 'responsibilities', 'issues', 'appraisal'].includes(k))
    }
    if (scopeLevel === 'department') {
      return ALL_TABS
    }
    return ALL_TABS
  }, [scopeLevel])

  useEffect(() => {
    if (!TABS.some(([k]) => k === tab)) setTab(TABS[0]?.[0] || 'dashboard')
  }, [TABS, tab])

  const scopeHint =
    scopeLevel === 'self'
      ? 'You see only your own responsibilities, one-time tasks, and appraisal.'
      : scopeLevel === 'department'
        ? 'You see your department team only.'
        : null

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">HRM — Task & Performance Tracker</h1>
        <p className="text-sm text-gray-500">
          Employees · Tasks · Issues · Dependency · Appraisal
          {scopeHint && <span className="block text-amber-700 mt-1">{scopeHint}</span>}
        </p>
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
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              ['Departments', (depts as any[]).length, 'text-blue-600'],
              ['Employees', (allEmps as any[]).length, 'text-purple-600'],
              ['Active', (allEmps as any[]).filter((e: any) => e.status === 'Active').length, 'text-green-600'],
              ['Departments', (depts as any[]).length, 'text-amber-600'],
            ].map(([l, v, c], i) => (
              <div key={i} className="bg-white rounded-xl p-4 border shadow-sm">
                <p className={`text-2xl font-bold ${c}`}>{v}</p>
                <p className="text-xs text-gray-500 mt-1 font-semibold">{l}</p>
              </div>
            ))}
          </div>

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
                  <div><span className="text-gray-500">Item Type:</span> <b>{aiParsed.item_type === 'task' ? 'Task' : 'Responsibility'}</b></div>
                  <div><span className="text-gray-500">Employee:</span> <b>{aiParsed.employee_name || '—'}</b></div>
                  <div className="col-span-2"><span className="text-gray-500">Title:</span> <b>{aiParsed.title || '—'}</b></div>
                  {aiParsed.item_type === 'responsibility' ? (
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
                  <select value={quickResp.item_type} onChange={e => setQuickResp(f => ({ ...f, item_type: e.target.value as ItemType }))}
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
                {quickResp.item_type === 'responsibility' ? (
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
                  </>
                ) : (
                  <div><label className="text-xs text-gray-500">Due Date</label>
                    <input type="date" value={quickResp.due_date} onChange={e => setQuickResp(f => ({ ...f, due_date: e.target.value }))}
                      className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                )}
                <div><label className="text-xs text-gray-500">Assigned By</label>
                  <input value={quickResp.added_by} onChange={e => setQuickResp(f => ({ ...f, added_by: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
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
                <div><label className="text-xs text-gray-500">Name *</label>
                  <input value={empForm.name} onChange={e => setEmpForm(f => ({ ...f, name: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
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
                <button onClick={() => createEmpMut.mutate({ ...empForm, department_id: empForm.department_id ? +empForm.department_id : null })} disabled={!empForm.name} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">Save</button>
                <button onClick={() => setShowEmpForm(false)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
              </div>
            </div>
          )}
          <div className="bg-white rounded-xl border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-xs text-gray-400 uppercase">
                <tr>{['Code','Name','Department','Designation','Phone','Status',''].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {(employees as any[]).map((e: any) => (
                  <tr key={e.id} className="border-t hover:bg-gray-50">
                    {editEmp?.id === e.id ? (
                      <td colSpan={7} className="px-4 py-3">
                        <div className="grid grid-cols-4 gap-2">
                          <input value={editEmp.name} onChange={ev => setEditEmp((x: any) => ({ ...x, name: ev.target.value }))} className="border rounded px-2 py-1 text-sm" />
                          <select value={editEmp.department_id || ''} onChange={ev => setEditEmp((x: any) => ({ ...x, department_id: +ev.target.value }))} className="border rounded px-2 py-1 text-sm">
                            <option value="">Dept</option>
                            {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
                          </select>
                          <input value={editEmp.designation} onChange={ev => setEditEmp((x: any) => ({ ...x, designation: ev.target.value }))} className="border rounded px-2 py-1 text-sm" placeholder="Designation" />
                          <div className="flex gap-1">
                            <button onClick={() => updateEmpMut.mutate({ id: e.id, data: editEmp })} className="px-2 py-1 bg-green-600 text-white rounded text-xs">Save</button>
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
                        <td className="px-4 py-2 text-gray-500">{e.phone || '—'}</td>
                        <td className="px-4 py-2"><span className={`text-xs px-2 py-0.5 rounded-full ${e.status === 'Active' ? 'bg-green-100 text-green-700' : 'bg-gray-100'}`}>{e.status}</span></td>
                        <td className="px-4 py-2">
                          <div className="flex gap-2">
                            <button onClick={() => setEditEmp({ id: e.id, name: e.name, department_id: e.department_id, designation: e.designation || '' })} className="text-xs text-blue-600">✏️</button>
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
              <select value={selDept} onChange={e => setSelDept(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Departments</option>
                {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
              </select>
              <select value={selEmp} onChange={e => setSelEmp(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Employees</option>
                {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
              </select>
            </div>
            <button onClick={() => setShowRespForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium">+ Assign Item</button>
          </div>
          {showRespForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">Assign Responsibility or Task</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div><label className="text-xs text-gray-500">Item Type *</label>
                  <select value={respForm.item_type} onChange={e => setRespForm(f => ({ ...f, item_type: e.target.value as ItemType }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="responsibility">Responsibility (recurring)</option>
                    <option value="task">Task (one-time)</option>
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">Employee *</label>
                  <select value={respForm.employee_id} onChange={e => setRespForm(f => ({ ...f, employee_id: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select</option>
                    {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name} ({e.department_name || '—'})</option>)}
                  </select>
                </div>
                <div className="col-span-2"><label className="text-xs text-gray-500">Title *</label>
                  <input value={respForm.title} onChange={e => setRespForm(f => ({ ...f, title: e.target.value }))} placeholder="e.g. Enter delivery date on bills" className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                {respForm.item_type === 'responsibility' ? (
                  <>
                    <div><label className="text-xs text-gray-500">Frequency</label>
                      <select value={respForm.frequency} onChange={e => setRespForm(f => ({ ...f, frequency: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                        {FREQUENCIES.map(f => <option key={f}>{f}</option>)}
                      </select>
                    </div>
                    <div><label className="text-xs text-gray-500">Category</label>
                      <select value={respForm.category} onChange={e => setRespForm(f => ({ ...f, category: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                        {CATEGORIES.map(c => <option key={c}>{c}</option>)}
                      </select>
                    </div>
                  </>
                ) : (
                  <div><label className="text-xs text-gray-500">Due Date</label>
                    <input type="date" value={respForm.due_date} onChange={e => setRespForm(f => ({ ...f, due_date: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                )}
                <div><label className="text-xs text-gray-500">Assigned By</label>
                  <input value={respForm.added_by} onChange={e => setRespForm(f => ({ ...f, added_by: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => submitAssign({ ...respForm, department_id: '' })} disabled={!respForm.employee_id || !respForm.title || createRespMut.isPending || createOneTimeTaskMut.isPending} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">Assign</button>
                <button onClick={() => setShowRespForm(false)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
              </div>
            </div>
          )}
          {(() => {
            const grouped: Record<string, any[]> = {}
            ;(responsibilities as any[]).forEach((r: any) => { const key = r.employee_name || 'Unknown'; if (!grouped[key]) grouped[key] = []; grouped[key].push(r) })
            return Object.entries(grouped).map(([empName, resps]) => (
              <div key={empName} className="bg-white rounded-xl border overflow-hidden">
                <div className="px-4 py-2 bg-[#002B5B] text-white text-sm font-semibold flex justify-between">
                  <span>👤 {empName}</span>
                  <span className="text-blue-200 text-xs">{resps[0]?.department_name || ''} · {resps.length} tasks</span>
                </div>
                <table className="w-full text-sm">
                  <thead className="text-gray-400 text-xs uppercase bg-gray-50">
                    <tr><th className="text-left px-4 py-2">Task</th><th className="text-left px-4 py-2">Frequency</th><th className="text-left px-4 py-2">Category</th><th className="text-left px-4 py-2">Added By</th><th className="px-4 py-2"></th></tr>
                  </thead>
                  <tbody>
                    {resps.map((r: any) => (
                      <tr key={r.id} className="border-t hover:bg-gray-50">
                        {editResp?.id === r.id ? (
                          <td colSpan={5} className="px-4 py-3">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                              <input value={editResp.title} onChange={e => setEditResp((x: any) => ({ ...x, title: e.target.value }))} className="border rounded px-2 py-1 text-sm col-span-2" placeholder="Title" />
                              <input value={editResp.description || ''} onChange={e => setEditResp((x: any) => ({ ...x, description: e.target.value }))} className="border rounded px-2 py-1 text-sm col-span-2" placeholder="Description" />
                              <select value={editResp.frequency} onChange={e => setEditResp((x: any) => ({ ...x, frequency: e.target.value }))} className="border rounded px-2 py-1 text-sm">
                                {FREQUENCIES.map(f => <option key={f}>{f}</option>)}
                              </select>
                              <select value={editResp.category} onChange={e => setEditResp((x: any) => ({ ...x, category: e.target.value }))} className="border rounded px-2 py-1 text-sm">
                                {CATEGORIES.map(c => <option key={c}>{c}</option>)}
                              </select>
                              <select value={editResp.employee_id} onChange={e => setEditResp((x: any) => ({ ...x, employee_id: +e.target.value }))} className="border rounded px-2 py-1 text-sm col-span-2">
                                {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
                              </select>
                              <div className="flex gap-2 col-span-2">
                                <button onClick={() => updateRespMut.mutate({ id: r.id, data: { title: editResp.title, description: editResp.description, frequency: editResp.frequency, category: editResp.category, employee_id: editResp.employee_id } })} disabled={!editResp.title || updateRespMut.isPending} className="px-3 py-1 bg-green-600 text-white rounded text-xs">Save</button>
                                <button onClick={() => setEditResp(null)} className="px-3 py-1 border rounded text-xs">Cancel</button>
                              </div>
                            </div>
                          </td>
                        ) : (
                          <>
                            <td className="px-4 py-2 font-medium">{r.title}{r.description && <p className="text-xs text-gray-400">{r.description}</p>}</td>
                            <td className="px-4 py-2"><span className={`text-xs px-2 py-0.5 rounded-full font-medium ${r.frequency === 'Daily' ? 'bg-blue-100 text-blue-700' : r.frequency === 'Weekly' ? 'bg-purple-100 text-purple-700' : 'bg-green-100 text-green-700'}`}>{r.frequency}</span></td>
                            <td className="px-4 py-2 text-xs text-gray-500">{r.category}</td>
                            <td className="px-4 py-2 text-xs text-gray-400">{r.added_by || '—'}</td>
                            <td className="px-4 py-2">
                              {canEditAssignments && (
                                <div className="flex gap-2">
                                  <button onClick={() => setEditResp({ id: r.id, title: r.title, description: r.description || '', frequency: r.frequency, category: r.category, employee_id: r.employee_id })} className="text-xs text-blue-600">✏️</button>
                                  <button onClick={() => { if (window.confirm('Remove?')) deleteRespMut.mutate(r.id) }} className="text-xs text-red-500">🗑️</button>
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
            ))
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
              <button onClick={() => setShowTaskForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium">+ Assign Task</button>
            )}
          </div>

          {showTaskForm && canAssignTasks && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">Assign One-Time Task</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div><label className="text-xs text-gray-500">Employee *</label>
                  <select value={taskForm.employee_id} onChange={e => setTaskForm(f => ({ ...f, employee_id: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select</option>
                    {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name} ({e.department_name || '—'})</option>)}
                  </select>
                </div>
                <div className="col-span-2"><label className="text-xs text-gray-500">Task Title *</label>
                  <input value={taskForm.title} onChange={e => setTaskForm(f => ({ ...f, title: e.target.value }))} placeholder="e.g. Complete warehouse audit by Friday" className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div className="col-span-2"><label className="text-xs text-gray-500">Description</label>
                  <input value={taskForm.description} onChange={e => setTaskForm(f => ({ ...f, description: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div><label className="text-xs text-gray-500">Due Date</label>
                  <input type="date" value={taskForm.due_date} onChange={e => setTaskForm(f => ({ ...f, due_date: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div><label className="text-xs text-gray-500">Assigned By</label>
                  <input value={taskForm.assigned_by} onChange={e => setTaskForm(f => ({ ...f, assigned_by: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => createOneTimeTaskMut.mutate({ ...taskForm, employee_id: +taskForm.employee_id })}
                  disabled={!taskForm.employee_id || !taskForm.title || createOneTimeTaskMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">
                  Assign
                </button>
                <button onClick={() => setShowTaskForm(false)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
              </div>
            </div>
          )}

          <div className="bg-white rounded-xl border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="text-gray-400 text-xs uppercase bg-gray-50">
                <tr>
                  <th className="text-left px-4 py-2">Task</th>
                  <th className="text-left px-4 py-2">Employee</th>
                  <th className="text-left px-4 py-2">Due</th>
                  <th className="text-left px-4 py-2">Status</th>
                  <th className="text-left px-4 py-2">Time</th>
                  <th className="px-4 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {(oneTimeTasks as any[]).map((t: any) => (
                  <tr key={t.id} className="border-t hover:bg-gray-50 align-top">
                    {editTask?.id === t.id ? (
                      <td colSpan={6} className="px-4 py-3">
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
                          {t.completion_notes && <p className="text-xs text-amber-700 mt-1">Done: {t.completion_notes}</p>}
                          {t.approval_notes && <p className="text-xs text-gray-500 mt-1">HOD: {t.approval_notes}</p>}
                        </td>
                        <td className="px-4 py-3">
                          <p className="font-medium">{t.employee_name}</p>
                          <p className="text-xs text-gray-400">{t.department_name || '—'}</p>
                          {t.assigned_by && <p className="text-xs text-gray-400 mt-0.5">By {t.assigned_by}</p>}
                        </td>
                        <td className="px-4 py-3 text-gray-600">{t.due_date || '—'}</td>
                        <td className="px-4 py-3">
                          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${oneTimeStatusStyle(t.status)}`}>{t.status}</span>
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-500">
                          <p>Start: {fmtDateTime(t.started_at)}</p>
                          <p>End: {fmtDateTime(t.completed_at)}</p>
                          <p className="font-semibold text-[#002B5B] mt-0.5">{fmtDuration(t.duration_minutes)}</p>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex flex-wrap gap-1 justify-end">
                            {canEditAssignments && t.status !== 'Approved' && (
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
                            {canAssignTasks && t.status !== 'Approved' && (
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
            {(oneTimeTasks as any[]).length === 0 && (
              <p className="text-center text-gray-400 py-8 text-sm">No one-time tasks yet.</p>
            )}
          </div>
        </div>
      )}

      {/* ── HOD VIEW ── */}
      {tab === 'hod' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            <select value={hodDept} onChange={e => setHodDept(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
              <option value="">Select Department</option>
              {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
            {hodSubTab === 'responsibilities' && (
              <>
                <input type="date" value={fromDate} onChange={e => setFromDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
                <span className="text-gray-400 text-xs">to</span>
                <input type="date" value={toDate} onChange={e => setToDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
              </>
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
          {hodDept && hodSubTab === 'responsibilities' && hodData && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 bg-[#002B5B] text-white flex justify-between">
                <span className="font-semibold">{deptName(hodDept)} — Task Dashboard</span>
                <div className="flex gap-3 text-xs text-blue-200">
                  <span>○ Pending</span><span>✅ Done</span><span>⚠️ Partial</span><span>❌ Missed</span><span>🔴 Blocked</span>
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
                        </td>
                        <td className="px-3 py-2 text-gray-400">{r.frequency}</td>
                        {(hodData.dates || []).map((d: string) => {
                          const dayData = r.dates?.[d] || { status: 'Pending' }
                          const s = dayData.status
                          return (
                            <td key={d} className="px-1 py-2 text-center">
                              <button
                                onClick={() => {
                                  const next = statusNext(s)
                                  if (next === 'Blocked') {
                                    setBlockedModal({ respId: r.id, date: d })
                                    setBlockedForm({ blocker_employee_id: '', blocker_reason: '', marked_by: '' })
                                  } else {
                                    markTaskMut.mutate({ responsibility_id: r.id, log_date: d, status: next })
                                  }
                                }}
                                title={`${s}${dayData.blocker_name ? ` — Blocked by ${dayData.blocker_name}` : ''}`}
                                className={`w-7 h-7 rounded-full text-xs font-bold transition-all hover:scale-110 ${statusBg(s)}`}>
                                {statusLabel(s)}
                              </button>
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
                Click to cycle: ○ → ✅ Done → ⚠️ Partial → ❌ Missed → 🔴 Blocked (cross-dept issue auto-create)
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── ISSUES ── */}
      {tab === 'issues' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex gap-2 flex-wrap">
              <select value={selDept} onChange={e => setSelDept(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Departments</option>
                {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
              </select>
              <input type="date" value={fromDate} onChange={e => setFromDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
              <input type="date" value={toDate} onChange={e => setToDate(e.target.value)} className="border rounded-lg px-3 py-1.5 text-sm" />
            </div>
            <button onClick={() => setShowIssueForm(true)} className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium">+ Record Issue</button>
          </div>

          {showIssueForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">⚠️ Record Issue / Problem</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div><label className="text-xs text-gray-500">Employee *</label>
                  <select value={issueForm.employee_id} onChange={e => setIssueForm(f => ({ ...f, employee_id: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select</option>
                    {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name} ({e.department_name || '—'})</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">Issue Type</label>
                  <select value={issueForm.issue_type} onChange={e => setIssueForm(f => ({ ...f, issue_type: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    {ISSUE_TYPES.map(t => <option key={t}>{t}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">Severity</label>
                  <select value={issueForm.severity} onChange={e => setIssueForm(f => ({ ...f, severity: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    {SEVERITIES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
                <div className="col-span-2"><label className="text-xs text-gray-500">Issue Title *</label>
                  <input value={issueForm.title} onChange={e => setIssueForm(f => ({ ...f, title: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div><label className="text-xs text-gray-500">Recorded By</label>
                  <input value={issueForm.recorded_by} onChange={e => setIssueForm(f => ({ ...f, recorded_by: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div className="col-span-3"><label className="text-xs text-gray-500">Description</label>
                  <input value={issueForm.description} onChange={e => setIssueForm(f => ({ ...f, description: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div><label className="text-xs text-gray-500">Caused By (cross-dept)</label>
                  <select value={issueForm.caused_by_employee_id} onChange={e => setIssueForm(f => ({ ...f, caused_by_employee_id: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">None</option>
                    {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name} ({e.department_name || '—'})</option>)}
                  </select>
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createIssueMut.mutate({ ...issueForm, employee_id: +issueForm.employee_id, caused_by_employee_id: issueForm.caused_by_employee_id ? +issueForm.caused_by_employee_id : null })}
                  disabled={!issueForm.employee_id || !issueForm.title}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm disabled:opacity-50">Record Issue</button>
                <button onClick={() => setShowIssueForm(false)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {(issues as any[]).map((issue: any) => (
              <div key={issue.id} className={`bg-white rounded-xl border p-4 border-l-4 ${issue.severity === 'Major' ? 'border-l-red-500' : issue.severity === 'Moderate' ? 'border-l-amber-500' : 'border-l-yellow-300'}`}>
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-gray-800">{issue.employee_name}</span>
                      <span className="text-xs text-gray-400">{issue.department_name}</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${issue.severity === 'Major' ? 'bg-red-100 text-red-700' : issue.severity === 'Moderate' ? 'bg-amber-100 text-amber-700' : 'bg-yellow-100 text-yellow-700'}`}>{issue.severity}</span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">{issue.issue_type}</span>
                    </div>
                    <p className="text-sm font-medium text-gray-700">{issue.title}</p>
                    {issue.description && <p className="text-xs text-gray-500 mt-0.5">{issue.description}</p>}
                    {issue.caused_by_name && (
                      <p className="text-xs text-purple-600 mt-1">🔗 Caused by: <b>{issue.caused_by_name}</b> ({issue.caused_by_dept_name})</p>
                    )}
                    <p className="text-xs text-gray-400 mt-1">{issue.issue_date} · By: {issue.recorded_by || '—'}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full ${issue.status === 'Resolved' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>{issue.status}</span>
                    {issue.status === 'Open' && (
                      <button onClick={() => { const res = prompt('Resolution:'); if (res) resolveIssueMut.mutate({ id: issue.id, res }) }}
                        className="text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700">Resolve</button>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {(issues as any[]).length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No issues found.</p>}
          </div>
        </div>
      )}

      {/* ── APPRAISAL ── */}
      {tab === 'appraisal' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            <select value={appraisalEmp} onChange={e => setAppraisalEmp(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
              <option value="">Select Employee</option>
              {(allEmps as any[]).map((e: any) => <option key={e.id} value={e.id}>{e.name} — {e.department_name || '—'}</option>)}
            </select>
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
          <div className="flex items-center gap-3 flex-wrap">
            <select value={selDept} onChange={e => setSelDept(e.target.value ? +e.target.value : '')} className="border rounded-lg px-3 py-1.5 text-sm">
              <option value="">All Departments</option>
              {(depts as any[]).map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
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
