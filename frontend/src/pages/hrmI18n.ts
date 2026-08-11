/** HRM bilingual strings — English / Hindi */

export type HrmLang = 'en' | 'hi'

const DICT = {
  en: {
    title: 'HRM — Task & Performance Tracker',
    subtitle: 'Employees · Tasks · Issues · Hierarchy · Appraisal',
    dashboard: 'Dashboard',
    check: 'Employee Check',
    employees: 'Employees',
    responsibilities: 'Responsibilities',
    tasks: 'Tasks',
    hod: 'HOD View',
    issues: 'Issues',
    appraisal: 'Appraisal',
    performance: 'Reports',
    hierarchy: 'Hierarchy',
    lang: 'Language',
    assignedBy: 'Assigned By',
    assignedTo: 'Assigned To',
    priority: 'Priority',
    mandatory: 'Mandatory',
    timePeriod: 'Time Period',
    frequency: 'Frequency',
    weekday: 'Weekday',
    monthDay: 'Day of month',
    filters: 'Filters',
    totalEmployees: 'Total Employees',
    departments: 'Departments',
    hodInfo: 'HOD',
    yes: 'Yes',
    no: 'No',
    save: 'Save',
    cancel: 'Cancel',
    edit: 'Edit',
    delete: 'Delete',
    manualTime: 'Manual time (HH:MM or minutes)',
    dueDate: 'Due Date',
    status: 'Status',
    completion: 'Completion %',
    taskReport: 'Task Report',
    voiceRecord: 'Voice record',
    playback: 'Playback',
    attachAudio: 'Attach audio',
    scheduleNote: 'Weekly/Monthly items appear in Employee Check only on the scheduled day.',
  },
  hi: {
    title: 'एचआरएम — कार्य और प्रदर्शन ट्रैकर',
    subtitle: 'कर्मचारी · कार्य · मुद्दे · पदानुक्रम · मूल्यांकन',
    dashboard: 'डैशबोर्ड',
    check: 'कर्मचारी चेक',
    employees: 'कर्मचारी',
    responsibilities: 'जिम्मेदारियाँ',
    tasks: 'कार्य',
    hod: 'एचओडी दृश्य',
    issues: 'मुद्दे',
    appraisal: 'मूल्यांकन',
    performance: 'रिपोर्ट',
    hierarchy: 'पदानुक्रम',
    lang: 'भाषा',
    assignedBy: 'आवंटितकर्ता',
    assignedTo: 'आवंटित',
    priority: 'प्राथमिकता',
    mandatory: 'अनिवार्य',
    timePeriod: 'समय अवधि',
    frequency: 'आवृत्ति',
    weekday: 'सप्ताह का दिन',
    monthDay: 'महीने का दिन',
    filters: 'फ़िल्टर',
    totalEmployees: 'कुल कर्मचारी',
    departments: 'विभाग',
    hodInfo: 'एचओडी',
    yes: 'हाँ',
    no: 'नहीं',
    save: 'सहेजें',
    cancel: 'रद्द',
    edit: 'संपादित',
    delete: 'हटाएँ',
    manualTime: 'मैन्युअल समय (HH:MM या मिनट)',
    dueDate: 'नियत तिथि',
    status: 'स्थिति',
    completion: 'पूर्णता %',
    taskReport: 'कार्य रिपोर्ट',
    voiceRecord: 'वॉइस रिकॉर्ड',
    playback: 'प्लेबैक',
    attachAudio: 'ऑडियो संलग्न करें',
    scheduleNote: 'साप्ताहिक/मासिक आइटम केवल निर्धारित दिन पर कर्मचारी चेक में दिखते हैं।',
  },
} as const

export type HrmDictKey = keyof typeof DICT.en

export function t(lang: HrmLang, key: HrmDictKey): string {
  return DICT[lang][key] || DICT.en[key] || key
}

export function loadHrmLang(): HrmLang {
  try {
    const v = localStorage.getItem('hrm_lang')
    if (v === 'hi' || v === 'en') return v
  } catch { /* ignore */ }
  return 'en'
}

export function saveHrmLang(lang: HrmLang) {
  try {
    localStorage.setItem('hrm_lang', lang)
  } catch { /* ignore */ }
}
