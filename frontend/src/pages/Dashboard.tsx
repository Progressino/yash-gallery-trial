import { useState, useMemo, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import api, {
  downloadDailyDsrCsv,
  downloadDsrBrandMonthlyCsv,
  downloadIntelligenceSalesCsv,
  getCoverage,
} from '../api/client'
import { useSession } from '../store/session'
import './Dashboard.css'

/* ═══════════════════════════════════════════════════════════
   TYPES
   ═══════════════════════════════════════════════════════════ */
interface DsrSection {
  platform: string
  rows: { segment: string; sales: number; returns: number }[]
  section_sales: number
  section_returns: number
}
interface DsrResponse {
  date: string
  display_date: string
  sections: DsrSection[]
  subtotal: { sales: number; returns: number }
}

interface DsrBrandMonthlyRow {
  month: string
  month_display: string
  YG: number
  Akiko: number
  Other: number
  Untagged: number
  leader: string
  delta: number
}
interface DsrBrandMonthlyResponse {
  rows: DsrBrandMonthlyRow[]
  totals: { YG: number; Akiko: number; Other: number; Untagged: number }
  note: string
}

interface PlatformSummaryItem {
  platform: string
  loaded: boolean
  total_units: number
  total_returns: number
  net_units?: number
  return_rate: number
  top_sku: string
  trend_direction: 'up' | 'down' | 'flat'
  trend_direction_net?: 'up' | 'down' | 'flat'
  monthly: { month: string; shipments: number; refunds: number; net?: number }[]
  by_state: { state: string; units: number; net_units?: number }[]
}
interface AnomalyItem {
  type: string
  severity: 'critical' | 'warning' | 'info'
  platform: string
  message: string
  sku?: string
}
interface SalesSummary {
  total_units: number
  total_returns: number
  net_units: number
  return_rate: number
  active_months?: number
  date_basis_note?: string
}
interface TopSku { sku: string; units: number }

/* ═══════════════════════════════════════════════════════════
   CONSTANTS
   ═══════════════════════════════════════════════════════════ */
const PLATFORM_COLORS: Record<string, string> = {
  Amazon:   'oklch(55% 0.15 260)',
  Myntra:   'oklch(62% 0.18 355)',
  Meesho:   'oklch(55% 0.17 310)',
  Flipkart: 'oklch(70% 0.17 65)',
  Snapdeal: 'oklch(60% 0.19 25)',
}
const PLATFORM_SHORT: Record<string, string> = {
  Amazon: 'AMZ', Myntra: 'MYN', Meesho: 'MEE', Flipkart: 'FK', Snapdeal: 'SD',
}

/* ═══════════════════════════════════════════════════════════
   DATE HELPERS
   ═══════════════════════════════════════════════════════════ */
function toIso(d: Date) { return d.toISOString().split('T')[0] }
function daysAgo(n: number) { const d = new Date(); d.setDate(d.getDate() - n); return toIso(d) }
function monthsAgo(n: number) { const d = new Date(); d.setMonth(d.getMonth() - n); return toIso(d) }
const TODAY = toIso(new Date())

type DatePreset = { label: string; start: () => string; end?: () => string }
const PRESETS: DatePreset[] = [
  { label: '30D',  start: () => daysAgo(30) },
  { label: '90D',  start: () => daysAgo(90) },
  { label: '6M',   start: () => monthsAgo(6) },
  { label: '1Y',   start: () => monthsAgo(12) },
  { label: 'All',  start: () => '' },
]

function fmtMonth(m: string) {
  try {
    const [y, mon] = m.split('-')
    return new Date(+y, +mon - 1).toLocaleString('default', { month: 'short', year: '2-digit' })
  } catch { return m }
}
function monthlyRowNet(row: PlatformSummaryItem['monthly'][0]) {
  return typeof row.net === 'number' ? row.net : row.shipments - row.refunds
}

/* ═══════════════════════════════════════════════════════════
   ICONS  (lucide-style, 16×16 rendered, 24 viewBox, 1.75 stroke)
   ═══════════════════════════════════════════════════════════ */
type SvgP = React.SVGProps<SVGSVGElement>
const svgBase: SvgP = {
  width: 16, height: 16, viewBox: '0 0 24 24',
  fill: 'none', stroke: 'currentColor',
  strokeWidth: 1.75, strokeLinecap: 'round', strokeLinejoin: 'round',
} as const

const Icon = {
  arrowUp:   (p?: SvgP) => <svg {...svgBase} {...p}><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>,
  arrowDown: (p?: SvgP) => <svg {...svgBase} {...p}><path d="M12 5v14"/><path d="M19 12l-7 7-7-7"/></svg>,
  arrowR:    (p?: SvgP) => <svg {...svgBase} {...p}><path d="M5 12h14"/><path d="M12 5l7 7-7 7"/></svg>,
  download:  (p?: SvgP) => <svg {...svgBase} {...p}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg>,
  upload:    (p?: SvgP) => <svg {...svgBase} {...p}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5-5 5 5"/><path d="M12 5v12"/></svg>,
  refresh:   (p?: SvgP) => <svg {...svgBase} {...p}><path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/></svg>,
  check:     (p?: SvgP) => <svg {...svgBase} {...p}><path d="M5 13l4 4L19 7"/></svg>,
  x:         (p?: SvgP) => <svg {...svgBase} {...p}><path d="M18 6L6 18"/><path d="M6 6l12 12"/></svg>,
  plus:      (p?: SvgP) => <svg {...svgBase} {...p}><path d="M12 5v14"/><path d="M5 12h14"/></svg>,
  more:      (p?: SvgP) => <svg {...svgBase} {...p}><circle cx="5" cy="12" r="1.2"/><circle cx="12" cy="12" r="1.2"/><circle cx="19" cy="12" r="1.2"/></svg>,
  filter:    (p?: SvgP) => <svg {...svgBase} {...p}><path d="M3 5h18l-7 9v6l-4-2v-4L3 5z"/></svg>,
  search:    (p?: SvgP) => <svg {...svgBase} {...p}><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>,
  chevR:     (p?: SvgP) => <svg {...svgBase} {...p}><path d="M9 6l6 6-6 6"/></svg>,
  chevD:     (p?: SvgP) => <svg {...svgBase} {...p}><path d="M6 9l6 6 6-6"/></svg>,
  alert:     (p?: SvgP) => <svg {...svgBase} {...p}><path d="M10.3 3.7l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3.3l-8-14a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>,
  info:      (p?: SvgP) => <svg {...svgBase} {...p}><circle cx="12" cy="12" r="9"/><path d="M12 16v-5"/><path d="M12 8h.01"/></svg>,
  sparkles:  (p?: SvgP) => <svg {...svgBase} {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/></svg>,
  sun:       (p?: SvgP) => <svg {...svgBase} {...p}><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>,
  moon:      (p?: SvgP) => <svg {...svgBase} {...p}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>,
  sliders:   (p?: SvgP) => <svg {...svgBase} {...p}><path d="M4 21v-7"/><path d="M4 10V3"/><path d="M12 21v-9"/><path d="M12 8V3"/><path d="M20 21v-5"/><path d="M20 12V3"/><circle cx="4" cy="12" r="2"/><circle cx="12" cy="10" r="2"/><circle cx="20" cy="14" r="2"/></svg>,
  trendUp:   (p?: SvgP) => <svg {...svgBase} {...p}><path d="M22 7l-10 10-5-5-5 5"/><path d="M16 7h6v6"/></svg>,
}

/* ═══════════════════════════════════════════════════════════
   PRIMITIVES
   ═══════════════════════════════════════════════════════════ */

function CountUp({ value, format = (v: number) => Math.round(v).toLocaleString() }: {
  value: number; format?: (v: number) => string
}) {
  const [n, setN] = useState(0)
  const ref = useRef<{ start: number; from: number }>({ start: 0, from: 0 })
  useEffect(() => {
    ref.current.from = n
    ref.current.start = performance.now()
    let raf: number
    const step = (t: number) => {
      const p = Math.min(1, (t - ref.current.start) / 900)
      const eased = 1 - Math.pow(1 - p, 3)
      setN(ref.current.from + (value - ref.current.from) * eased)
      if (p < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [value]) // eslint-disable-line react-hooks/exhaustive-deps
  return <span>{format(n)}</span>
}

function BigSpark({ values, color, height = 56, width = 220 }: {
  values: number[]; color: string; height?: number; width?: number
}) {
  if (!values || values.length < 2) return null
  const max = Math.max(...values)
  const min = Math.min(...values)
  const span = max - min || 1
  const step = width / (values.length - 1)
  const pts = values.map((v, i) => [i * step, height - ((v - min) / span) * (height - 8) - 4] as [number, number])
  let smooth = `M${pts[0][0]},${pts[0][1]}`
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] ?? pts[i]
    const p1 = pts[i]
    const p2 = pts[i + 1]
    const p3 = pts[i + 2] ?? p2
    const cp1x = p1[0] + (p2[0] - p0[0]) / 6
    const cp1y = p1[1] + (p2[1] - p0[1]) / 6
    const cp2x = p2[0] - (p3[0] - p1[0]) / 6
    const cp2y = p2[1] - (p3[1] - p1[1]) / 6
    smooth += ` C${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`
  }
  const area = `${smooth} L${width},${height} L0,${height} Z`
  const gid = 'g' + Math.random().toString(36).slice(2, 7)
  const last = pts[pts.length - 1]
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', overflow: 'visible' }}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gid})`} />
      <path d={smooth} stroke={color} strokeWidth="1.75" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={last[0]} cy={last[1]} r="4" fill={color} opacity="0.2" />
      <circle cx={last[0]} cy={last[1]} r="2.5" fill={color} />
    </svg>
  )
}

function RadialGauge({ value, max = 100, color = 'var(--primary)', label, size = 110 }: {
  value: number; max?: number; color?: string; label?: string; size?: number
}) {
  const r = (size - 16) / 2
  const c = 2 * Math.PI * r
  const pct = Math.min(1, value / max)
  const arcLen = c * 0.75
  const dash = arcLen * pct
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ display: 'block' }}>
      <g transform={`rotate(135 ${size / 2} ${size / 2})`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none"
          stroke="var(--border)" strokeWidth="6"
          strokeDasharray={`${arcLen} ${c}`} strokeLinecap="round" />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none"
          stroke={color} strokeWidth="6"
          strokeDasharray={`${dash} ${c}`} strokeLinecap="round"
          style={{ transition: 'stroke-dasharray .9s cubic-bezier(.2,.8,.2,1)' }} />
      </g>
      <text x={size / 2} y={size / 2 - 3} textAnchor="middle" fontSize="19" fontWeight="700"
        fill="var(--ink)" fontFamily="var(--font-sans)" style={{ fontVariantNumeric: 'tabular-nums' }}>
        {Math.round(value)}%
      </text>
      {label && (
        <text x={size / 2} y={size / 2 + 13} textAnchor="middle" fontSize="9"
          fill="var(--muted)" fontFamily="var(--font-sans)" letterSpacing="0.06em">
          {label.toUpperCase()}
        </text>
      )}
    </svg>
  )
}

/* ═══════════════════════════════════════════════════════════
   HERO CHART
   ═══════════════════════════════════════════════════════════ */
interface HeroSeries { name: string; color: string; values: number[] }

function HeroChart({ series, months, hidden, viewMode }: {
  series: HeroSeries[]; months: string[]; hidden: Set<string>; viewMode: string
}) {
  const W = 820, H = 280
  const pad = { t: 24, r: 28, b: 32, l: 0 }
  const plotW = W - pad.l - pad.r
  const plotH = H - pad.t - pad.b
  const visible = series.filter(s => !hidden.has(s.name))
  const maxV = Math.max(1, ...visible.flatMap(s => s.values)) * 1.18
  const stepX = months.length > 1 ? plotW / (months.length - 1) : plotW
  const [hover, setHover] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const totalsByMonth = months.map((_, i) =>
    visible.reduce((s, sr) => s + (sr.values[i] ?? 0), 0)
  )

  const pathFor = (vals: number[]) => {
    const pts = vals.map((v, i) => [pad.l + i * stepX, pad.t + plotH - (v / maxV) * plotH] as [number, number])
    let d = `M${pts[0][0]},${pts[0][1]}`
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] ?? pts[i]
      const p1 = pts[i]
      const p2 = pts[i + 1]
      const p3 = pts[i + 2] ?? p2
      const cp1x = p1[0] + (p2[0] - p0[0]) / 6
      const cp1y = p1[1] + (p2[1] - p0[1]) / 6
      const cp2x = p2[0] - (p3[0] - p1[0]) / 6
      const cp2y = p2[1] - (p3[1] - p1[1]) / 6
      d += ` C${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`
    }
    return d
  }
  const areaFor = (vals: number[]) =>
    `${pathFor(vals)} L${pad.l + plotW},${pad.t + plotH} L${pad.l},${pad.t + plotH} Z`

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current || months.length < 2) return
    const rect = svgRef.current.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * W - pad.l
    const i = Math.round(x / stepX)
    if (i >= 0 && i < months.length) setHover(i)
  }

  return (
    <div style={{ position: 'relative' }}>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} width="100%"
        style={{ display: 'block', overflow: 'visible' }}
        onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        <defs>
          {visible.map(s => (
            <linearGradient key={s.name} id={`hcg-${s.name.replace(/\s/g,'')}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity="0.22" />
              <stop offset="100%" stopColor={s.color} stopOpacity="0" />
            </linearGradient>
          ))}
        </defs>
        {/* y grid */}
        {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
          const y = pad.t + plotH - ratio * plotH
          return (
            <g key={i}>
              <line x1={pad.l} x2={W - pad.r} y1={y} y2={y}
                stroke="var(--border)" strokeDasharray="1 3" opacity="0.7" />
              <text x={W - pad.r + 5} y={y + 3.5} fontSize="9.5" fill="var(--muted)" fontFamily="var(--font-mono)">
                {(ratio * maxV / 1000).toFixed(0)}k
              </text>
            </g>
          )
        })}
        {/* x labels */}
        {months.map((m, i) => (
          <text key={i} x={pad.l + i * stepX} y={H - 6} fontSize="10.5"
            fill={hover === i ? 'var(--ink)' : 'var(--muted)'}
            fontWeight={hover === i ? 600 : 400}
            textAnchor="middle">{m}</text>
        ))}
        {/* areas */}
        {visible.map(s => (
          <path key={'a' + s.name} d={areaFor(s.values)}
            fill={`url(#hcg-${s.name.replace(/\s/g,'')})`} />
        ))}
        {/* lines */}
        {visible.map(s => (
          <path key={'l' + s.name} d={pathFor(s.values)}
            stroke={s.color} strokeWidth="2.2" fill="none"
            strokeLinecap="round" strokeLinejoin="round"
            style={{ filter: `drop-shadow(0 2px 5px ${s.color}44)` }} />
        ))}
        {/* dots */}
        {visible.map(s => s.values.map((v, i) => {
          const x = pad.l + i * stepX
          const y = pad.t + plotH - (v / maxV) * plotH
          return (
            <circle key={s.name + i} cx={x} cy={y} r={hover === i ? 5 : 3}
              fill="var(--surface)" stroke={s.color}
              strokeWidth={hover === i ? 2.5 : 1.75}
              style={{ transition: 'r .12s' }} />
          )
        }))}
        {/* hover rail */}
        {hover !== null && (
          <line x1={pad.l + hover * stepX} x2={pad.l + hover * stepX}
            y1={pad.t} y2={pad.t + plotH}
            stroke="var(--ink)" strokeOpacity="0.12" strokeDasharray="2 3" />
        )}
      </svg>

      {/* Tooltip */}
      {hover !== null && (
        <div style={{
          position: 'absolute',
          left: `calc(${((pad.l + hover * stepX) / W) * 100}% + 14px)`,
          top: 8,
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 10,
          padding: '10px 14px',
          boxShadow: 'var(--shadow-md)',
          minWidth: 155,
          pointerEvents: 'none',
          zIndex: 10,
        }}>
          <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.09em', marginBottom: 5 }}>
            {months[hover]}
          </div>
          <div style={{ fontSize: 20, fontWeight: 800, fontVariantNumeric: 'tabular-nums', color: 'var(--ink)', lineHeight: 1 }}>
            {totalsByMonth[hover].toLocaleString()}
          </div>
          <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 8 }}>total {viewMode}</div>
          {visible.map(s => (
            <div key={s.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, marginTop: 4 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10.5, color: 'var(--ink-soft)' }}>
                <span style={{ width: 7, height: 7, background: s.color, borderRadius: 2, flexShrink: 0 }} />
                {s.name}
              </span>
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--ink)' }}>
                {(s.values[hover] ?? 0).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════
   HERO KPI
   ═══════════════════════════════════════════════════════════ */
function HeroKPI({ eyebrow, value, unit, delta, deltaDir, caption, spark, color }: {
  eyebrow: string; value: number; unit?: string
  delta?: string; deltaDir?: 'up' | 'down'
  caption?: string; spark?: number[]; color?: string
}) {
  const col = color ?? 'var(--primary)'
  return (
    <div className="hk">
      <div className="hk-eyebrow">{eyebrow}</div>
      <div className="hk-row">
        <div className="hk-value">
          <CountUp value={value} />
          {unit && <span className="hk-unit">{unit}</span>}
        </div>
        {delta && deltaDir && (
          <div className={`hk-delta ${deltaDir}`}>
            {deltaDir === 'up' ? <Icon.arrowUp /> : <Icon.arrowDown />}
            <span>{delta}</span>
          </div>
        )}
      </div>
      <div className="hk-foot">
        {caption && <span className="hk-caption">{caption}</span>}
        {spark && spark.length > 1 && (
          <div className="hk-spark">
            <BigSpark values={spark} color={col} width={100} height={26} />
          </div>
        )}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════
   INDIA HEAT MAP
   ═══════════════════════════════════════════════════════════ */
const INDIA_GRID: [number, number, string][] = [
  [0, 4, 'Punjab'], [0, 5, 'Haryana'], [0, 6, 'Delhi'], [0, 7, 'Uttar Pradesh'],
  [1, 3, 'Rajasthan'], [1, 6, 'Uttar Pradesh'], [1, 8, 'Bihar'],
  [2, 2, 'Gujarat'], [2, 4, 'Madhya Pradesh'], [2, 8, 'West Bengal'], [2, 9, 'Odisha'],
  [3, 2, 'Maharashtra'], [3, 4, 'Maharashtra'], [3, 6, 'Telangana'], [3, 8, 'Odisha'],
  [4, 3, 'Karnataka'], [4, 5, 'Andhra Pradesh'],
  [5, 3, 'Kerala'], [5, 5, 'Tamil Nadu'],
]

function IndiaHeat({ states }: { states: { state: string; units: number }[] }) {
  const map = Object.fromEntries(states.map(s => [s.state, s.units]))
  const max = Math.max(...states.map(s => s.units), 1)
  const cellSize = 30, gap = 3
  const cols = 12, rows = 7

  const intensity = (units: number) => {
    if (!units) return { bg: 'var(--bg-sunken)', fg: 'var(--muted)' }
    const r = units / max
    if (r > 0.8) return { bg: 'oklch(42% 0.16 255)', fg: 'white' }
    if (r > 0.6) return { bg: 'oklch(55% 0.16 255)', fg: 'white' }
    if (r > 0.4) return { bg: 'oklch(68% 0.14 255)', fg: 'white' }
    if (r > 0.2) return { bg: 'oklch(82% 0.09 255)', fg: 'oklch(30% 0.15 255)' }
    return { bg: 'oklch(92% 0.04 255)', fg: 'oklch(40% 0.15 255)' }
  }

  const sorted = [...states].sort((a, b) => b.units - a.units).slice(0, 5)

  return (
    <div className="india-wrap">
      <svg viewBox={`0 0 ${cols * (cellSize + gap)} ${rows * (cellSize + gap)}`}
        width="100%" style={{ display: 'block', maxHeight: 230 }}>
        {INDIA_GRID.map(([r, c, state], i) => {
          const units = map[state] ?? 0
          const col = intensity(units)
          return (
            <g key={i}>
              <rect x={c * (cellSize + gap)} y={r * (cellSize + gap)}
                width={cellSize} height={cellSize} rx="5"
                fill={col.bg} style={{ transition: 'fill .4s ease' }}>
                <title>{state}: {units.toLocaleString()}</title>
              </rect>
              <text x={c * (cellSize + gap) + cellSize / 2}
                y={r * (cellSize + gap) + cellSize / 2 + 3}
                textAnchor="middle" fontSize="8" fontWeight="600"
                fill={col.fg} fontFamily="var(--font-mono)">
                {state.slice(0, 2).toUpperCase()}
              </text>
            </g>
          )
        })}
      </svg>

      <div className="india-legend">
        <span>Lower</span>
        {[0.05, 0.3, 0.5, 0.7, 0.9].map(r => {
          const col = intensity(r * max)
          return <span key={r} className="leg-box" style={{ background: col.bg }} />
        })}
        <span>Higher</span>
      </div>

      {sorted.length > 0 && (
        <div className="india-top">
          {sorted.map((s, i) => (
            <div key={s.state} className="india-top-row">
              <span className="india-rank">{String(i + 1).padStart(2, '0')}</span>
              <span className="india-name">{s.state}</span>
              <div className="india-bar">
                <div className="india-bar-fill" style={{
                  width: `${(s.units / max) * 100}%`,
                  background: intensity(s.units).bg,
                }} />
              </div>
              <span className="india-val">{s.units.toLocaleString()}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════
   PLATFORM TILE
   ═══════════════════════════════════════════════════════════ */
function PlatformTile({ p, salesViewNet, onClick }: {
  p: PlatformSummaryItem
  salesViewNet: boolean
  onClick: () => void
}) {
  const color = PLATFORM_COLORS[p.platform] ?? '#6366F1'
  const short = PLATFORM_SHORT[p.platform] ?? p.platform.slice(0, 3).toUpperCase()
  const tDir = salesViewNet ? (p.trend_direction_net ?? p.trend_direction) : p.trend_direction
  const units = salesViewNet ? (p.net_units ?? p.total_units - p.total_returns) : p.total_units
  const monthly = p.monthly.map(r => salesViewNet ? monthlyRowNet(r) : r.shipments).filter(v => v > 0)
  const rr = p.return_rate

  return (
    <div className={`ptile-2 ${!p.loaded ? 'off' : ''}`}
      style={{ '--p-col': color } as React.CSSProperties}
      onClick={p.loaded ? onClick : undefined}>
      <div className="ptile-2-glow" />
      <div className="ptile-2-head">
        <div className="ptile-2-logo">{short}</div>
        <div className="ptile-2-name">
          <div className="ptile-2-brand">{p.platform}</div>
          <div className="ptile-2-sub">{p.loaded ? 'loaded' : 'not loaded'}</div>
        </div>
        {p.loaded && (
          <div className={`trend-pill ${tDir}`}>
            {tDir === 'up' ? <Icon.arrowUp /> : tDir === 'down' ? <Icon.arrowDown /> : null}
          </div>
        )}
      </div>
      <div className="ptile-2-value">{p.loaded ? units.toLocaleString() : '—'}</div>
      <div className="ptile-2-foot">
        {p.loaded ? (
          <>
            <span className={`badge ${rr > 30 ? 'danger' : rr > 15 ? 'warn' : 'success'}`}>
              {rr.toFixed(1)}% return
            </span>
            {monthly.length > 1 && (
              <BigSpark values={monthly} color={color} width={80} height={22} />
            )}
          </>
        ) : (
          <span className="badge neutral">offline</span>
        )}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════
   SKU BAR LIST (replaces matrix since no per-platform SKU data)
   ═══════════════════════════════════════════════════════════ */
function SkuList({ skus, platforms, salesViewNet, onOpenSku }: {
  skus: TopSku[]
  platforms: PlatformSummaryItem[]
  salesViewNet: boolean
  onOpenSku: (sku: string) => void
}) {
  const activePlatforms = platforms.filter(p => p.loaded).map(p => ({
    id: p.platform.toLowerCase(),
    name: p.platform,
    color: PLATFORM_COLORS[p.platform] ?? '#6366F1',
    short: PLATFORM_SHORT[p.platform] ?? p.platform.slice(0, 3),
  }))
  const max = Math.max(...skus.map(s => s.units), 1)

  return (
    <div className="matrix">
      <div className="matrix-head" style={{ gridTemplateColumns: `1fr ${activePlatforms.map(() => '54px').join(' ')} 64px` }}>
        <div>SKU</div>
        {activePlatforms.map(p => (
          <div key={p.id} className="mh-plat">
            <span className="mh-dot" style={{ background: p.color }} />
            {p.short}
          </div>
        ))}
        <div style={{ textAlign: 'right' }}>Units</div>
      </div>

      {skus.map((s, idx) => {
        const barPct = (s.units / max) * 100
        return (
          <div key={s.sku} className="matrix-row"
            style={{ gridTemplateColumns: `1fr ${activePlatforms.map(() => '54px').join(' ')} 64px` }}
            onClick={() => onOpenSku(s.sku)}>
            <div className="matrix-sku">
              <div className="mx-sku-code">{s.sku}</div>
              <div className="mx-sku-name" style={{
                width: `${Math.max(10, barPct)}%`,
                height: 3,
                background: `oklch(${60 - idx * 3}% 0.14 255)`,
                borderRadius: 99,
                marginTop: 4,
              }} />
            </div>
            {activePlatforms.map(p => (
              <div key={p.id} className="matrix-cell" style={{
                background: 'var(--bg-sunken)',
                color: 'var(--muted)',
              }}>—</div>
            ))}
            <div className="matrix-total">
              {salesViewNet ? '~' : ''}{s.units.toLocaleString()}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════
   BRAND STACKS
   ═══════════════════════════════════════════════════════════ */
function BrandStacks({ data }: { data: DsrBrandMonthlyRow[] }) {
  const maxTotal = Math.max(...data.map(r => r.YG + r.Akiko + (r.Other ?? 0)), 1)
  const totalYG = data.reduce((s, r) => s + r.YG, 0)
  const totalAK = data.reduce((s, r) => s + r.Akiko, 0)

  return (
    <div>
      <div style={{ display: 'flex', gap: 20, marginBottom: 16, flexWrap: 'wrap' }}>
        <div>
          <div className="brand-total-eyebrow">YG</div>
          <div className="brand-total-val">{totalYG.toLocaleString()}</div>
          <div className="brand-total-bar">
            <div className="brand-total-fill"
              style={{ width: `${totalYG / (totalYG + totalAK || 1) * 100}%`, background: 'var(--primary)' }} />
          </div>
        </div>
        <div>
          <div className="brand-total-eyebrow" style={{ color: 'var(--p-myntra)' }}>AKIKO</div>
          <div className="brand-total-val">{totalAK.toLocaleString()}</div>
          <div className="brand-total-bar">
            <div className="brand-total-fill"
              style={{ width: `${totalAK / (totalYG + totalAK || 1) * 100}%`, background: 'var(--p-myntra)' }} />
          </div>
        </div>
      </div>

      <div className="brand-months">
        {data.map((r, i) => {
          const total = r.YG + r.Akiko
          const ygPct = r.YG / maxTotal * 100
          const akPct = r.Akiko / maxTotal * 100
          return (
            <div key={i} className="brand-month">
              <div className="brand-month-label">{r.month_display}</div>
              <div className="brand-month-bar">
                <div className="brand-month-yg" style={{ height: `${ygPct}%` }} />
                <div className="brand-month-ak" style={{ height: `${akPct}%` }} />
              </div>
              <div className="brand-month-total">{(total / 1000).toFixed(1)}k</div>
              <div className={`brand-month-leader ${r.leader === 'YG' ? 'yg' : 'ak'}`}>
                {r.leader === 'YG' ? '◆' : '●'}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════
   DSR TABLE
   ═══════════════════════════════════════════════════════════ */
function DsrTable({ data }: { data: DsrSection[] }) {
  const totalSales = data.reduce((s, sec) => s + sec.rows.reduce((x, r) => x + r.sales, 0), 0)
  const totalRet   = data.reduce((s, sec) => s + sec.rows.reduce((x, r) => x + r.returns, 0), 0)
  return (
    <table className="tbl dsr-tbl">
      <thead>
        <tr>
          <th>Channel</th>
          <th>Segment</th>
          <th className="num">Sales</th>
          <th className="num">Returns</th>
          <th className="num">Rate</th>
          <th>Share</th>
        </tr>
      </thead>
      <tbody>
        {data.map(sec => {
          const color = PLATFORM_COLORS[sec.platform] ?? '#6366F1'
          const short = PLATFORM_SHORT[sec.platform] ?? sec.platform.slice(0, 3)
          return sec.rows.map((r, i) => {
            const rr = (r.returns / r.sales * 100).toFixed(1)
            const share = r.sales / totalSales * 100
            return (
              <tr key={`${sec.platform}-${r.segment}`}>
                <td>
                  {i === 0 ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{
                        width: 20, height: 20, borderRadius: 5, background: color,
                        color: 'white', fontSize: 8.5, fontFamily: 'var(--font-mono)',
                        fontWeight: 800, display: 'grid', placeItems: 'center', flexShrink: 0,
                      }}>{short}</span>
                      <span style={{ fontWeight: 600, fontSize: 12 }}>{sec.platform}</span>
                    </div>
                  ) : <span style={{ color: 'var(--muted)' }}>—</span>}
                </td>
                <td style={{ color: 'var(--ink-soft)', fontSize: 11.5 }}>{r.segment}</td>
                <td className="num" style={{ fontWeight: 600 }}>{r.sales.toLocaleString()}</td>
                <td className="num" style={{ color: r.returns > 30 ? 'var(--danger)' : 'var(--muted)' }}>
                  {r.returns.toLocaleString()}
                </td>
                <td className="num">
                  <span className={`badge ${Number(rr) > 25 ? 'danger' : Number(rr) > 15 ? 'warn' : 'success'}`}>
                    {rr}%
                  </span>
                </td>
                <td>
                  <div className="share-bar">
                    <div className="share-fill" style={{ width: `${Math.min(100, share * 3)}%`, background: color }} />
                  </div>
                </td>
              </tr>
            )
          })
        })}
        <tr className="dsr-total">
          <td colSpan={2} style={{ fontWeight: 700 }}>TOTAL</td>
          <td className="num" style={{ fontWeight: 700 }}>{totalSales.toLocaleString()}</td>
          <td className="num" style={{ fontWeight: 700, color: 'var(--danger)' }}>{totalRet.toLocaleString()}</td>
          <td className="num" style={{ fontWeight: 700 }}>
            {totalSales > 0 ? (totalRet / totalSales * 100).toFixed(1) : '0.0'}%
          </td>
          <td />
        </tr>
      </tbody>
    </table>
  )
}

/* ═══════════════════════════════════════════════════════════
   MAIN DASHBOARD
   ═══════════════════════════════════════════════════════════ */
export default function Dashboard() {
  const navigate = useNavigate()
  const setCoverage = useSession(s => s.setCoverage)
  const salesLoaded = useSession(s => s.sales)

  /* ── state ── */
  const [dateStart,      setDateStart]      = useState(() => daysAgo(90))
  const [dateEnd,        setDateEnd]        = useState(TODAY)
  const [activePreset,   setActivePreset]   = useState('90D')
  const [salesViewNet,   setSalesViewNet]   = useState(false)
  const [hiddenPlatforms,setHiddenPlatforms]= useState<Set<string>>(new Set())
  const [heatPlatform,   setHeatPlatform]   = useState('Myntra')
  const [topSkuLimit,    setTopSkuLimit]    = useState(10)
  const [showDsr,        setShowDsr]        = useState(false)
  const [dsrDate,        setDsrDate]        = useState(TODAY)
  const [exportingSales,     setExportingSales]     = useState(false)
  const [exportingDsr,       setExportingDsr]       = useState(false)
  const [exportingDsrMonthly,setExportingDsrMonthly]= useState(false)
  const [skuSearch,      setSkuSearch]      = useState('')

  /* ── preset ── */
  function applyPreset(label: string, startFn: () => string, endFn?: () => string) {
    setDateStart(startFn())
    setDateEnd(endFn ? endFn() : toIso(new Date()))
    setActivePreset(label)
  }
  function togglePlatform(name: string) {
    setHiddenPlatforms(prev => {
      const n = new Set(prev); n.has(name) ? n.delete(name) : n.add(name); return n
    })
  }

  /* ── query params ── */
  const summaryParams = useMemo(() => {
    const p = new URLSearchParams({ months: '0' })
    if (dateStart) p.set('start_date', dateStart)
    if (dateEnd)   p.set('end_date',   dateEnd)
    return p.toString()
  }, [dateStart, dateEnd])

  const dateParams = useMemo(() => {
    const p = new URLSearchParams({ limit: String(topSkuLimit) })
    if (dateStart) p.set('start_date', dateStart)
    if (dateEnd)   p.set('end_date',   dateEnd)
    if (salesViewNet) p.set('basis', 'net')
    return p.toString()
  }, [dateStart, dateEnd, topSkuLimit, salesViewNet])

  const dsrBrandMonthlyParams = useMemo(() => {
    const p = new URLSearchParams()
    if (dateStart) p.set('start_date', dateStart)
    if (dateEnd)   p.set('end_date', dateEnd)
    return p.toString()
  }, [dateStart, dateEnd])

  /* ── queries ── */
  useQuery({
    queryKey: ['coverage'],
    queryFn: async () => { const c = await getCoverage(); setCoverage(c); return c },
    staleTime: 60_000, refetchInterval: 120_000,
  })
  const { data: salesSummary } = useQuery<SalesSummary>({
    queryKey: ['sales-summary', dateStart, dateEnd],
    queryFn: async () => { const { data } = await api.get(`/data/sales-summary?${summaryParams}`); return data },
    staleTime: 120_000,
  })
  const { data: topSkusRaw } = useQuery<TopSku[]>({
    queryKey: ['top-skus', dateStart, dateEnd, topSkuLimit, salesViewNet],
    queryFn: async () => { const { data } = await api.get(`/data/top-skus?${dateParams}`); return data },
    staleTime: 120_000,
  })
  const { data: platformSummary, isLoading: loadingPlatforms } = useQuery<PlatformSummaryItem[]>({
    queryKey: ['platform-summary', dateStart, dateEnd],
    queryFn: async () => { const { data } = await api.get(`/data/platform-summary?${summaryParams}`); return data },
    staleTime: 300_000,
  })
  const { data: anomalies } = useQuery<AnomalyItem[]>({
    queryKey: ['anomalies', dateStart, dateEnd],
    queryFn: async () => { const { data } = await api.get(`/data/anomalies?${summaryParams}`); return data },
    staleTime: 120_000,
  })
  const { data: dsrData, isLoading: loadingDsr } = useQuery<DsrResponse>({
    queryKey: ['daily-dsr', dsrDate],
    queryFn: async () => { const { data } = await api.get(`/data/daily-dsr?date=${encodeURIComponent(dsrDate)}`); return data },
    enabled: showDsr && !!dsrDate,
    staleTime: 60_000,
  })
  const { data: dsrBrandMonthly, isLoading: loadingDsrBrands } = useQuery<DsrBrandMonthlyResponse>({
    queryKey: ['dsr-brand-monthly', dateStart, dateEnd],
    queryFn: async () => { const { data } = await api.get(`/data/dsr-brand-monthly?${dsrBrandMonthlyParams}`); return data },
    enabled: salesLoaded,
    staleTime: 60_000,
  })

  /* ── derived ── */
  const platforms = platformSummary ?? []
  const loadedPlatforms = platforms.filter(p => p.loaded)

  const filteredPlatforms = useMemo(() => {
    const startM = dateStart.slice(0, 7)
    const endM   = dateEnd.slice(0, 7)
    return platforms.map(p => ({
      ...p,
      monthly: p.monthly.filter(r =>
        (!startM || r.month >= startM) && (!endM || r.month <= endM)
      ),
    }))
  }, [platforms, dateStart, dateEnd])

  const allMonths = useMemo(() => {
    const months = new Set<string>()
    for (const p of filteredPlatforms) {
      if (!p.loaded) continue
      for (const r of p.monthly) months.add(r.month)
    }
    return Array.from(months).sort()
  }, [filteredPlatforms])

  const monthLabels = useMemo(() => allMonths.map(fmtMonth), [allMonths])

  const platformSeries = useMemo<HeroSeries[]>(() => {
    return filteredPlatforms
      .filter(p => p.loaded && !hiddenPlatforms.has(p.platform))
      .map(p => {
        const mm: Record<string, number> = {}
        for (const r of p.monthly) mm[r.month] = salesViewNet ? monthlyRowNet(r) : r.shipments
        return {
          name: p.platform,
          color: PLATFORM_COLORS[p.platform] ?? '#6366F1',
          values: allMonths.map(m => mm[m] ?? 0),
        }
      })
  }, [filteredPlatforms, hiddenPlatforms, allMonths, salesViewNet])

  const allMonthlyTotals = useMemo(() =>
    allMonths.map((_, i) => platformSeries.reduce((s, p) => s + p.values[i], 0)),
    [allMonths, platformSeries]
  )

  const totalUnits  = salesSummary?.total_units  ?? 0
  const totalReturns= salesSummary?.total_returns ?? 0
  const netUnits    = salesSummary?.net_units     ?? 0
  const returnRate  = salesSummary?.return_rate   ?? 0
  const displayUnits= salesViewNet ? netUnits : totalUnits

  const heatmapData = useMemo(() => {
    const p = filteredPlatforms.find(x => x.platform === heatPlatform)
    if (!p) return []
    return p.by_state.map(s => ({ state: s.state, units: salesViewNet ? (s.net_units ?? s.units) : s.units }))
  }, [filteredPlatforms, heatPlatform, salesViewNet])

  const topSkusFiltered = useMemo(() => {
    if (!topSkusRaw) return []
    if (!skuSearch.trim()) return topSkusRaw
    const q = skuSearch.trim().toLowerCase()
    return topSkusRaw.filter(s => s.sku.toLowerCase().includes(q))
  }, [topSkusRaw, skuSearch])

  /* ── forecast (simple extrapolation) ── */
  const forecastMonths = ['May 26', 'Jun 26', 'Jul 26']
  const lastTotal = allMonthlyTotals[allMonthlyTotals.length - 1] ?? 0
  const withForecast = useMemo<HeroSeries[]>(() => {
    if (!lastTotal) return platformSeries
    return platformSeries.map(s => {
      const last = s.values[s.values.length - 1] ?? 0
      return {
        ...s,
        values: [...s.values, Math.round(last * 1.06), Math.round(last * 1.11), Math.round(last * 1.18)],
      }
    })
  }, [platformSeries, lastTotal])

  /* ── exports ── */
  const exportPlatforms = useMemo(() => {
    if (hiddenPlatforms.size === 0) return undefined
    const names = loadedPlatforms.filter(p => !hiddenPlatforms.has(p.platform)).map(p => p.platform)
    return names.length ? names : undefined
  }, [loadedPlatforms, hiddenPlatforms])

  async function handleDownloadSalesCsv() {
    try { setExportingSales(true); await downloadIntelligenceSalesCsv({ startDate: dateStart || undefined, endDate: dateEnd || undefined, platforms: exportPlatforms }) }
    catch (e) { window.alert(e instanceof Error ? e.message : 'Download failed') }
    finally { setExportingSales(false) }
  }
  async function handleDownloadDsrCsv() {
    try { setExportingDsr(true); await downloadDailyDsrCsv(dsrDate) }
    catch (e) { window.alert(e instanceof Error ? e.message : 'Export failed') }
    finally { setExportingDsr(false) }
  }
  async function handleDownloadDsrMonthlyCsv() {
    try { setExportingDsrMonthly(true); await downloadDsrBrandMonthlyCsv(dsrBrandMonthlyParams) }
    catch (e) { window.alert(e instanceof Error ? e.message : 'Export failed') }
    finally { setExportingDsrMonthly(false) }
  }

  const hiddenByName = new Set([...hiddenPlatforms].map(id => id))

  /* ────────────────────────────────────────────────────────────────
     RENDER
     ──────────────────────────────────────────────────────────────── */
  return (
    <div className="dash-v2">

      {/* ══════════ HERO ══════════ */}
      <section className="hero">
        <div className="hero-bg" aria-hidden>
          <div className="hero-blob hero-blob-a" />
          <div className="hero-blob hero-blob-b" />
          <div className="hero-grid" />
        </div>

        <div className="hero-head">
          <div>
            <div className="hero-eyebrow">
              <span className="hero-pulse" />
              LIVE · INTELLIGENCE CONSOLE
            </div>
            <h1 className="hero-title">
              Every unit. Every channel.<br />
              <span className="hero-title-accent">One view.</span>
            </h1>
            <p className="hero-sub">
              {dateStart || 'All time'} → {dateEnd || 'today'} ·{' '}
              {loadedPlatforms.length} of {platforms.length || 5} marketplaces loaded
            </p>
          </div>
          <div className="hero-actions">
            <div className="seg">
              {PRESETS.map(({ label, start, end }) => (
                <button key={label} className={activePreset === label ? 'on' : ''}
                  onClick={() => applyPreset(label, start, end)}>{label}</button>
              ))}
            </div>
            <div className="seg">
              <button className={!salesViewNet ? 'on' : ''} onClick={() => setSalesViewNet(false)}>Gross</button>
              <button className={salesViewNet ? 'on' : ''}  onClick={() => setSalesViewNet(true)}>Net</button>
            </div>
            <button className="btn sm" onClick={() => void handleDownloadSalesCsv()}
              disabled={!salesLoaded || exportingSales}>
              <Icon.download />{exportingSales ? 'Preparing…' : 'Export'}
            </button>
          </div>
        </div>

        <div className="hero-grid-layout">
          {/* Giant number */}
          <div className="hero-total">
            <div className="hero-total-eyebrow">
              TOTAL {salesViewNet ? 'NET' : 'GROSS'} UNITS · {activePreset || 'CUSTOM RANGE'}
            </div>
            <div className="hero-total-value">
              {loadingPlatforms
                ? <span style={{ opacity: 0.4 }}>—</span>
                : <CountUp value={displayUnits} />}
            </div>
            <div className="hero-total-sub">
              {returnRate > 0 && (
                <span className={`trend-pill ${returnRate < 20 ? 'up' : 'down'}`}>
                  {returnRate < 20 ? <Icon.arrowUp /> : <Icon.arrowDown />}
                  {returnRate.toFixed(1)}% return rate
                </span>
              )}
              <span className="sep">·</span>
              <span>₹{(displayUnits * 680 / 10_000_000).toFixed(1)}Cr est. GMV</span>
            </div>
            {allMonthlyTotals.length > 1 && (
              <div className="hero-total-spark">
                <BigSpark values={allMonthlyTotals} color="var(--primary)" width={360} height={60} />
              </div>
            )}
          </div>

          {/* Side KPIs */}
          <div className="hero-kpis">
            <HeroKPI
              eyebrow="RETURN RATE"
              value={parseFloat(returnRate.toFixed(1))}
              unit="%"
              deltaDir={returnRate < 20 ? 'down' : 'up'}
              caption="shipments vs returns"
              spark={allMonthlyTotals.length > 1 ? allMonthlyTotals.map(v => (v > 0 ? (totalReturns / totalUnits * v) : 0)) : undefined}
              color="var(--warn)"
            />
            <HeroKPI
              eyebrow="NET UNITS"
              value={netUnits}
              caption="after returns"
              spark={allMonthlyTotals.length > 1 ? allMonthlyTotals.map(v => Math.round(v * (1 - returnRate / 100))) : undefined}
              color="var(--success)"
            />
            <HeroKPI
              eyebrow="SIGNALS"
              value={anomalies?.length ?? 0}
              caption={`${(anomalies ?? []).filter(a => a.severity === 'critical').length} critical`}
              color="var(--danger)"
            />
          </div>
        </div>
      </section>

      {/* ══════════ CONTROLS ══════════ */}
      <div className="controls-card">
        {/* Date row */}
        <div className="controls-row">
          <span className="controls-label">Date</span>
          <div className="ctrl-preset-group">
            {PRESETS.map(({ label, start, end }) => (
              <button key={label} className={`ctrl-preset ${activePreset === label ? 'on' : ''}`}
                onClick={() => applyPreset(label, start, end)}>{label}</button>
            ))}
          </div>
          <div className="controls-row" style={{ gap: 6 }}>
            <input type="date" className="ctrl-date" value={dateStart} max={dateEnd || TODAY}
              onChange={e => { setDateStart(e.target.value); setActivePreset('') }} />
            <span className="ctrl-arrow">→</span>
            <input type="date" className="ctrl-date" value={dateEnd} min={dateStart} max={TODAY}
              onChange={e => { setDateEnd(e.target.value); setActivePreset('') }} />
          </div>
        </div>
        {salesSummary?.date_basis_note && (
          <div className="date-basis-note" style={{ marginTop: 8 }}>
            {salesSummary.date_basis_note}
          </div>
        )}
        <div className="controls-divider" />

        {/* Gross / Net + platforms + DSR */}
        <div className="controls-row" style={{ gap: 14 }}>
          <span className="controls-label">View</span>
          <div className="seg-light">
            <button className={!salesViewNet ? 'on' : ''} onClick={() => setSalesViewNet(false)}>Gross</button>
            <button className={salesViewNet ? 'on' : ''} onClick={() => setSalesViewNet(true)}>Net</button>
          </div>

          {loadedPlatforms.length > 0 && (
            <>
              <span className="controls-label" style={{ marginLeft: 4 }}>Platforms</span>
              <div className="plat-chip-group">
                {loadedPlatforms.map(p => {
                  const hidden = hiddenPlatforms.has(p.platform)
                  const color = PLATFORM_COLORS[p.platform] ?? '#6366F1'
                  return (
                    <button key={p.platform}
                      className={`plat-chip ${hidden ? 'hidden' : ''}`}
                      style={!hidden ? { background: color, color: '#fff', borderColor: color } : {}}
                      onClick={() => togglePlatform(p.platform)}>
                      {p.platform}
                    </button>
                  )
                })}
                {hiddenPlatforms.size > 0 && (
                  <button style={{ fontSize: 11, color: 'var(--primary)', background: 'none', border: 'none', cursor: 'pointer' }}
                    onClick={() => setHiddenPlatforms(new Set())}>Show all</button>
                )}
              </div>
            </>
          )}

          <label className="dsr-toggle-label" style={{ marginLeft: 'auto' }}>
            <input type="checkbox" checked={showDsr}
              onChange={e => { setShowDsr(e.target.checked); if (e.target.checked) setDsrDate(dateEnd || TODAY) }} />
            Daily DSR
          </label>
          {showDsr && (
            <input type="date" className="ctrl-date" value={dsrDate} max={TODAY}
              onChange={e => setDsrDate(e.target.value)} />
          )}
        </div>
      </div>

      {/* ══════════ DAILY DSR ══════════ */}
      {showDsr && (
        <div className="dash-section">
          <div className="card">
            <div className="card-head">
              <div>
                <div className="card-title">Daily Sales Report</div>
                <div className="card-sub">{dsrData?.display_date ?? dsrDate}</div>
              </div>
              <div className="card-actions">
                <button className="btn sm ghost" disabled={!salesLoaded || exportingDsr || loadingDsr}
                  onClick={handleDownloadDsrCsv}>
                  <Icon.download />{exportingDsr ? 'Exporting…' : 'Export CSV'}
                </button>
              </div>
            </div>
            <div className="card-body flush">
              {!salesLoaded ? (
                <p style={{ padding: '24px', textAlign: 'center', color: 'var(--muted)', fontSize: 13 }}>Load sales data to view DSR.</p>
              ) : loadingDsr ? (
                <p style={{ padding: '24px', textAlign: 'center', color: 'var(--muted)', fontSize: 13 }}>Loading…</p>
              ) : !dsrData?.sections?.length ? (
                <p style={{ padding: '24px', textAlign: 'center', color: 'var(--muted)', fontSize: 13 }}>
                  No data for this day — try a different date.
                </p>
              ) : (
                <DsrTable data={dsrData.sections} />
              )}
            </div>
          </div>
        </div>
      )}

      {/* ══════════ CHART + GAUGES ══════════ */}
      <div className="grid-2">
        <div className="card chart-card">
          <div className="card-head">
            <div>
              <div className="card-title">Sales velocity · {activePreset || 'custom range'}</div>
              <div className="card-sub">
                {salesViewNet ? 'Net' : 'Gross'} shipments by channel · forecast shaded
              </div>
            </div>
            <div className="card-actions">
              <div className="chip-group">
                {platforms.map(p => {
                  const active = p.loaded && !hiddenPlatforms.has(p.platform)
                  const col = PLATFORM_COLORS[p.platform] ?? '#6366F1'
                  return (
                    <button key={p.platform}
                      className={`chip ${active ? 'on' : ''}`}
                      style={active ? {
                        background: `color-mix(in oklch, ${col} 12%, var(--surface))`,
                        borderColor: `color-mix(in oklch, ${col} 40%, transparent)`,
                        color: col,
                      } : { opacity: p.loaded ? 0.45 : 0.3 }}
                      onClick={() => p.loaded && togglePlatform(p.platform)}
                      disabled={!p.loaded}>
                      <span className="chip-dot" style={{ background: col }} />
                      {p.platform}
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
          <div className="card-body">
            {loadingPlatforms ? (
              <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: 13 }}>
                Loading…
              </div>
            ) : withForecast.length === 0 || allMonths.length === 0 ? (
              <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: 13 }}>
                No data — <a href="/upload" style={{ color: 'var(--primary)', marginLeft: 4 }}>upload files</a>
              </div>
            ) : (
              <HeroChart
                series={withForecast}
                months={[...monthLabels, ...forecastMonths]}
                hidden={hiddenByName}
                viewMode={salesViewNet ? 'net' : 'gross'}
              />
            )}
          </div>
        </div>

        <div className="card gauge-card">
          <div className="card-head">
            <div>
              <div className="card-title">Health signals</div>
              <div className="card-sub">Platform health overview</div>
            </div>
          </div>
          <div className="card-body" style={{ paddingTop: 12 }}>
            <div className="gauges">
              <div className="gauge-item">
                <RadialGauge value={94} color="var(--success)" label="Fulfillment" />
              </div>
              <div className="gauge-item">
                <RadialGauge value={loadedPlatforms.length / Math.max(platforms.length, 1) * 100}
                  color="var(--primary)" label="Coverage" />
              </div>
              <div className="gauge-item">
                <RadialGauge value={Math.max(0, 100 - returnRate)}
                  color="var(--warn)" label="Keep rate" />
              </div>
              <div className="gauge-item">
                <RadialGauge value={68} color="var(--info)" label="Forecast conf." />
              </div>
            </div>
            <div className="gauge-note">
              <Icon.sparkles />
              <span>Analytics refreshed live · <span className="mono">data</span></span>
            </div>
          </div>
        </div>
      </div>

      {/* ══════════ PLATFORM RAIL ══════════ */}
      <div className="platform-rail-2">
        {(loadingPlatforms
          ? ['Amazon', 'Myntra', 'Meesho', 'Flipkart', 'Snapdeal'].map(name => ({
              platform: name, loaded: false, total_units: 0, total_returns: 0,
              return_rate: 0, top_sku: '', trend_direction: 'flat' as const,
              monthly: [], by_state: [],
            } as PlatformSummaryItem))
          : platforms
        ).map(p => (
          <PlatformTile key={p.platform} p={p} salesViewNet={salesViewNet}
            onClick={() => navigate('/upload')} />
        ))}
      </div>

      {/* ══════════ SKU LIST + ANOMALIES ══════════ */}
      <div className="grid-2b">
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">Top SKUs</div>
              <div className="card-sub">
                By {salesViewNet ? 'net' : 'gross'} units · click to open deepdive
              </div>
            </div>
            <div className="card-actions">
              <input type="text" placeholder="Search SKU…" value={skuSearch}
                onChange={e => setSkuSearch(e.target.value)}
                style={{
                  fontSize: 11, border: '1px solid var(--border)', borderRadius: 7,
                  padding: '4px 10px', outline: 'none', fontFamily: 'var(--font-sans)',
                  color: 'var(--ink)', background: 'var(--surface)',
                }} />
              <select value={topSkuLimit} onChange={e => setTopSkuLimit(Number(e.target.value))}
                style={{
                  fontSize: 11, border: '1px solid var(--border)', borderRadius: 7,
                  padding: '4px 8px', outline: 'none', fontFamily: 'var(--font-sans)',
                  color: 'var(--ink-soft)', background: 'var(--surface)',
                }}>
                {[10, 20, 30, 50].map(n => <option key={n} value={n}>Top {n}</option>)}
              </select>
            </div>
          </div>
          <div className="card-body flush">
            {topSkusFiltered.length === 0 ? (
              <p style={{ padding: '24px', textAlign: 'center', color: 'var(--muted)', fontSize: 13 }}>
                {skuSearch ? 'No matching SKUs' : 'No SKU data'}
              </p>
            ) : (
              <SkuList
                skus={topSkusFiltered.slice(0, 10)}
                platforms={filteredPlatforms}
                salesViewNet={salesViewNet}
                onOpenSku={sku => navigate(`/sku-deepdive?sku=${encodeURIComponent(sku)}`)}
              />
            )}
          </div>
        </div>

        <div className="card alerts-card">
          <div className="card-head">
            <div>
              <div className="card-title">Signals</div>
              <div className="card-sub">
                {(anomalies?.length ?? 0) > 0 ? `${anomalies!.length} active anomalies` : 'All clear'}
              </div>
            </div>
          </div>
          <div className="card-body flush">
            {!anomalies || anomalies.length === 0 ? (
              <div style={{ padding: '20px 16px', display: 'flex', gap: 12, alignItems: 'center' }}>
                <span style={{ fontSize: 24 }}>✅</span>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--ink)' }}>All Clear</div>
                  <div style={{ fontSize: 11, color: 'var(--muted)' }}>No anomalies detected</div>
                </div>
              </div>
            ) : (
              anomalies.map((a, i) => (
                <div key={i} className={`alert-2 ${a.severity}`}
                  onClick={() => a.sku && navigate(`/sku-deepdive?sku=${encodeURIComponent(a.sku)}`)}
                  style={{ cursor: a.sku ? 'pointer' : 'default' }}>
                  <div className="alert-2-sev">
                    {a.severity === 'critical' ? '!!' : a.severity === 'warning' ? '!' : 'i'}
                  </div>
                  <div className="alert-2-body">
                    <div className="alert-2-head">
                      <span className="alert-2-title">{a.type}</span>
                      <span className="alert-2-plat"
                        style={{ color: PLATFORM_COLORS[a.platform] ?? 'var(--muted)' }}>
                        {a.platform}
                      </span>
                    </div>
                    <div className="alert-2-desc">{a.message}</div>
                    {a.sku && <span className="alert-2-sku">→ {a.sku}</span>}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* ══════════ GEOGRAPHY + BRAND ══════════ */}
      <div className="grid-2">
        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">Geography · India</div>
              <div className="card-sub">Units shipped by state · {salesViewNet ? 'net' : 'gross'}</div>
            </div>
            <div className="card-actions">
              <div className="seg-light">
                {loadedPlatforms.map(p => (
                  <button key={p.platform}
                    className={heatPlatform === p.platform ? 'on' : ''}
                    onClick={() => setHeatPlatform(p.platform)}>
                    {p.platform}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="card-body">
            {heatmapData.length > 0 ? (
              <IndiaHeat states={heatmapData} />
            ) : (
              <p style={{ color: 'var(--muted)', fontSize: 13, textAlign: 'center', padding: '24px 0' }}>
                No state data available
              </p>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <div>
              <div className="card-title">Brand split · YG vs Akiko</div>
              <div className="card-sub">Monthly shipments · {salesViewNet ? 'net' : 'gross'}</div>
            </div>
            <div className="card-actions">
              <button className="btn sm ghost" disabled={exportingDsrMonthly || loadingDsrBrands}
                onClick={handleDownloadDsrMonthlyCsv}>
                <Icon.download />{exportingDsrMonthly ? 'Exporting…' : 'CSV'}
              </button>
            </div>
          </div>
          <div className="card-body">
            {loadingDsrBrands ? (
              <p style={{ color: 'var(--muted)', fontSize: 13 }}>Loading…</p>
            ) : !dsrBrandMonthly?.rows?.length ? (
              <p style={{ color: 'var(--muted)', fontSize: 13 }}>
                {dsrBrandMonthly?.note || 'No brand data in range'}
              </p>
            ) : (
              <BrandStacks data={dsrBrandMonthly.rows} />
            )}
          </div>
        </div>
      </div>

      {/* ══════════ BRAND BAR CHART (full table) ══════════ */}
      {salesLoaded && dsrBrandMonthly?.rows?.length && (
        <div className="dash-section">
          <div className="card">
            <div className="card-head">
              <div>
                <div className="card-title">YG vs Akiko vs Other vs Untagged — monthly</div>
                <div className="card-sub">
                  All named sellers · four columns sum to gross shipments for the selected range
                </div>
              </div>
            </div>
            <div className="card-body">
              <div style={{ height: 220 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dsrBrandMonthly.rows.map(r => ({
                    name: r.month_display, YG: r.YG, Akiko: r.Akiko,
                    Other: r.Other, Untagged: r.Untagged ?? 0,
                  }))} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
                    <Tooltip formatter={(v: number | undefined) => (v ?? 0).toLocaleString()} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="YG"       fill="#002B5B" radius={[2,2,0,0]} />
                    <Bar dataKey="Akiko"    fill="#E91E63" radius={[2,2,0,0]} />
                    <Bar dataKey="Other"    fill="#94A3B8" radius={[2,2,0,0]} />
                    <Bar dataKey="Untagged" fill="#F97316" radius={[2,2,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
