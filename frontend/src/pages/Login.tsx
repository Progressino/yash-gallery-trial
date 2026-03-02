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

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await api.post('/auth/login', { username, password })
      nav('/', { replace: true })
    } catch {
      setError('Invalid username or password')
    } finally {
      setLoading(false)
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
                <input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  placeholder="••••••••"
                  className="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[#002B5B] focus:border-transparent shadow-sm transition"
                />
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
                    Signing in…
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
