/** Lightweight usage tracking — page views, tab switches and an "active" heartbeat.
 *  Sends counts only (path / module / tab name), never page contents. */
import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import api from '../api/client'
import { moduleForPath } from './modules'

type UsageEvent = { kind: 'page' | 'tab'; path: string; module: string; tab?: string }

const FLUSH_MS = 60_000
const IDLE_MS = 2 * 60_000
const MAX_BATCH = 50

let queue: UsageEvent[] = []
let enabled = false
let started = false
let lastInput = Date.now()
let lastPage = ''
let lastPageAt = 0

function moduleOf(path: string): string {
  return moduleForPath(path) ?? ''
}

function userIsActive(): boolean {
  return document.visibilityState === 'visible' && Date.now() - lastInput < IDLE_MS
}

export function trackUsage(ev: UsageEvent) {
  if (!enabled) return
  queue.push(ev)
  if (queue.length >= MAX_BATCH) void flushUsage()
}

export async function flushUsage(beacon = false) {
  if (!enabled) return
  const events = queue.splice(0, MAX_BATCH)
  const active = userIsActive()
  if (!events.length && !active) return
  const body = { events, active }
  if (beacon && typeof navigator.sendBeacon === 'function') {
    navigator.sendBeacon('/api/usage/track', new Blob([JSON.stringify(body)], { type: 'application/json' }))
    return
  }
  try {
    await api.post('/usage/track', body, { timeout: 15_000 })
  } catch {
    /* best effort */
  }
}

function startListeners() {
  if (started) return
  started = true
  const mark = () => { lastInput = Date.now() }
  const opts: AddEventListenerOptions = { passive: true, capture: true }
  for (const ev of ['pointerdown', 'keydown', 'wheel', 'touchstart', 'scroll']) {
    window.addEventListener(ev, mark, opts)
  }
  window.setInterval(() => void flushUsage(), FLUSH_MS)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') void flushUsage(true)
  })
  window.addEventListener('pagehide', () => void flushUsage(true))
}

/** Mount once inside the router for signed-in users. */
export function useUsageTracking(on: boolean) {
  const { pathname } = useLocation()

  useEffect(() => {
    enabled = on
    if (on) startListeners()
    else queue = []
  }, [on])

  useEffect(() => {
    if (!on) return
    const now = Date.now()
    if (pathname === lastPage && now - lastPageAt < 3_000) return
    lastPage = pathname
    lastPageAt = now
    lastInput = now
    trackUsage({ kind: 'page', path: pathname, module: moduleOf(pathname) })
  }, [on, pathname])
}

/** Record which tab inside a page is opened (tabs that are not part of the URL). */
export function useTrackTab(tab: string | null | undefined) {
  const { pathname } = useLocation()
  useEffect(() => {
    if (!tab) return
    trackUsage({ kind: 'tab', path: pathname, module: moduleOf(pathname), tab: String(tab) })
  }, [tab, pathname])
}
