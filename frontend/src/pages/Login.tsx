import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../api/client'
export default function Login() {
  const nav = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [loadStep, setLoadStep] = useState('')
  const [showPwd, setShowPwd]   = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      setLoadStep('Signing in…')
      await api.post('/auth/login', { username, password })
      nav('/', { replace: true })
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number } })?.response?.status
      if (status === 401) {
        setError('Invalid username or password')
      } else {
        nav('/', { replace: true })
      }
    } finally {
      setLoading(false)
      setLoadStep('')
    }
  }

  return (
    <div className="min-h-screen flex">
      {/* ── Left branding panel ── */}
      <div className="hidden lg:flex lg:w-1/2 flex-col bg-[#002B5B] relative overflow-hidden">
        {/* Subtle background circles */}
        <div className="absolute -top-24 -left-24 w-96 h-96 rounded-full bg-white/5" />
        <div className="absolute -bottom-32 -right-16 w-[28rem] h-[28rem] rounded-full bg-white/5" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 rounded-full bg-white/[0.03]" />

        {/* Centered content */}
        <div className="flex-1 flex flex-col items-center justify-center px-16 relative z-10">
          {/* Logo on white pill */}
          <div className="bg-white rounded-2xl px-8 py-6 shadow-2xl mb-10">
            <img src="/logo.png" alt="Progressino" className="h-20 w-auto" />
          </div>

          <h1 className="text-white text-3xl font-bold text-center leading-snug">
            Yash Gallery<br />
            <span className="text-blue-300 font-normal text-xl">ERP Command Center</span>
          </h1>

          <p className="text-blue-200/70 text-sm mt-5 text-center max-w-xs leading-relaxed">
            Complete e-commerce intelligence — sales analytics, inventory,
            PO engine &amp; AI forecasting across all platforms.
          </p>

          {/* Feature chips */}
          <div className="flex flex-wrap gap-2 mt-8 justify-center">
            {['📊 Analytics', '📦 Inventory', '🎯 PO Engine', '📈 AI Forecast'].map(f => (
              <span key={f} className="bg-white/10 text-white/80 text-xs px-3 py-1 rounded-full border border-white/10">
                {f}
              </span>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="py-6 flex items-center justify-center gap-2 relative z-10">
          <img src="/logo.png" alt="" className="h-5 w-auto opacity-60 mix-blend-luminosity" />
          <span className="text-blue-200/50 text-xs">Built by Progressino © 2025</span>
        </div>
      </div>

      {/* ── Right login panel ── */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {/* Mobile logo bar */}
        <div className="lg:hidden flex items-center gap-3 px-6 py-5 bg-[#002B5B]">
          <div className="bg-white rounded-lg px-3 py-1.5">
            <img src="/logo.png" alt="Progressino" className="h-7 w-auto" />
          </div>
          <span className="text-white font-semibold text-sm">Yash Gallery ERP</span>
        </div>

        <div className="flex-1 flex items-center justify-center p-8">
          <div className="w-full max-w-sm">
            {/* Heading */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900">Welcome back</h2>
              <p className="text-gray-500 text-sm mt-1">Sign in to your workspace</p>
            </div>

            <form onSubmit={submit} className="space-y-5">
              <div>
                <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">
                  Username
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  required
                  autoFocus
                  placeholder="admin"
                  className="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[#002B5B] focus:border-transparent shadow-sm transition"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">
                  Password
                </label>
                <div className="relative">
                  <input
                    type={showPwd ? 'text' : 'password'}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    required
                    placeholder="••••••••"
                    className="w-full border border-gray-200 rounded-xl px-4 py-3 pr-11 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[#002B5B] focus:border-transparent shadow-sm transition"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPwd(v => !v)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                    tabIndex={-1}
                    aria-label={showPwd ? 'Hide password' : 'Show password'}
                  >
                    {showPwd ? (
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
                      </svg>
                    ) : (
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>

              {error && (
                <div className="flex items-center gap-2 bg-red-50 border border-red-100 rounded-xl px-4 py-3">
                  <span className="text-red-500 text-xs">⚠</span>
                  <p className="text-red-600 text-xs font-medium">{error}</p>
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-[#002B5B] hover:bg-[#003875] active:scale-[0.98] disabled:opacity-50 shadow-md shadow-blue-900/20 transition-all mt-2"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                    </svg>
                    {loadStep || 'Signing in…'}
                  </span>
                ) : 'Sign In'}
              </button>
            </form>

            {/* Mobile "Built by" */}
            <p className="lg:hidden text-center text-gray-400 text-xs mt-10">
              Built by{' '}
              <span className="font-semibold text-gray-500">Progressino</span>
              {' '}© 2025
            </p>
          </div>
        </div>

        {/* Desktop footer */}
        <div className="hidden lg:flex items-center justify-center py-5 text-gray-400 text-xs gap-1">
          Built by <span className="font-semibold text-gray-500 ml-1">Progressino</span>
        </div>
      </div>
    </div>
  )
}
