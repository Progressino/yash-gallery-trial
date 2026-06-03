import { useState } from 'react'
import type { FormEvent } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import api from '../api/client'
import { useAuth, type AuthUser } from '../store/auth'
import { getDeviceId } from '../lib/deviceId'

type Step = 'credentials' | 'otp'

function profileFromLoginData(data: Record<string, unknown>): AuthUser {
  return {
    username: String(data.username ?? ''),
    role: String(data.role ?? ''),
    full_name: String(data.full_name ?? ''),
    karigar_id: String(data.karigar_id ?? ''),
    user_id: data.user_id as number | undefined,
    employee_id: data.employee_id as number | null | undefined,
    hrm_department_id: data.hrm_department_id as number | null | undefined,
    modules: data.modules as string[] | undefined,
    hrm_scope: data.hrm_scope as AuthUser['hrm_scope'],
    is_karigar: data.role === 'Karigar',
    may_upload_historical: data.may_upload_historical as boolean | undefined,
    may_reset_all: data.may_reset_all as boolean | undefined,
  }
}

export default function Login() {
  const setUser = useAuth(s => s.setUser)
  const nav = useNavigate()
  const location = useLocation()
  const serverUnreachable = !!(location.state as { serverUnreachable?: boolean } | null)?.serverUnreachable
  const [step, setStep] = useState<Step>('credentials')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [otp, setOtp] = useState('')
  const [challengeId, setChallengeId] = useState('')
  const [maskedPhone, setMaskedPhone] = useState('')
  const [trustDevice, setTrustDevice] = useState(true)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadStep, setLoadStep] = useState('')
  const [showPwd, setShowPwd] = useState(false)

  const deviceHeaders = () => ({ 'X-Device-Id': getDeviceId() })

  const finishLogin = (data: Record<string, unknown>) => {
    const profile = profileFromLoginData(data)
    setUser(profile)
    const dest = (data.redirect as string) || (data.role === 'Karigar' ? '/production-entry' : '/')
    nav(dest, { replace: true })
  }

  const submitCredentials = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      setLoadStep('Signing in…')
      const { data } = await api.post(
        '/auth/login',
        { username, password },
        { timeout: 90_000, headers: deviceHeaders() },
      )
      if (data?.otp_required && data?.challenge_id) {
        setChallengeId(data.challenge_id)
        setMaskedPhone(data.masked_phone || '')
        setStep('otp')
        setOtp('')
        return
      }
      finishLogin(data)
    } catch (err: unknown) {
      const e = err as { response?: { status?: number; data?: { detail?: string } }; code?: string; message?: string }
      const status = e?.response?.status
      const detail = e?.response?.data?.detail
      if (status === 401) setError('Invalid username or password')
      else if (status === 503 && detail) setError(detail)
      else if (status === 502 || status === 504) {
        setError('Server gateway timeout. Wait a minute and try again.')
      } else if (e?.code === 'ECONNABORTED' || /timeout/i.test(e?.message || '')) {
        setError('Server is taking too long. Wait a minute and try again.')
      } else if (!e?.response) {
        setError('Cannot reach the server. Check your connection.')
      } else {
        setError(typeof detail === 'string' ? detail : 'Sign-in failed. Please try again.')
      }
    } finally {
      setLoading(false)
      setLoadStep('')
    }
  }

  const submitOtp = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      setLoadStep('Verifying OTP…')
      const { data } = await api.post(
        '/auth/otp/verify',
        { challenge_id: challengeId, code: otp.trim(), trust_device: trustDevice },
        { timeout: 60_000, headers: deviceHeaders() },
      )
      finishLogin(data)
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } } }
      setError(e?.response?.data?.detail || 'Invalid OTP. Try again.')
    } finally {
      setLoading(false)
      setLoadStep('')
    }
  }

  const resendOtp = async () => {
    setError('')
    setLoading(true)
    try {
      const { data } = await api.post('/auth/otp/resend', { challenge_id: challengeId }, { headers: deviceHeaders() })
      setMaskedPhone(data.masked_phone || maskedPhone)
      setLoadStep('OTP sent again')
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } } }
      setError(e?.response?.data?.detail || 'Could not resend OTP')
    } finally {
      setLoading(false)
      setTimeout(() => setLoadStep(''), 2000)
    }
  }

  const backToCredentials = () => {
    setStep('credentials')
    setOtp('')
    setChallengeId('')
    setError('')
  }

  return (
    <div className="min-h-screen flex">
      <div className="hidden lg:flex lg:w-1/2 flex-col bg-[#002B5B] relative overflow-hidden">
        <div className="absolute -top-24 -left-24 w-96 h-96 rounded-full bg-white/5" />
        <div className="absolute -bottom-32 -right-16 w-[28rem] h-[28rem] rounded-full bg-white/5" />
        <div className="flex-1 flex flex-col items-center justify-center px-16 relative z-10">
          <div className="bg-white rounded-2xl px-8 py-6 shadow-2xl mb-10">
            <img src="/logo.png" alt="Progressino" className="h-20 w-auto" />
          </div>
          <h1 className="text-white text-3xl font-bold text-center leading-snug">
            Yash Gallery<br />
            <span className="text-blue-300 font-normal text-xl">ERP Command Center</span>
          </h1>
          <p className="text-blue-200/70 text-sm mt-5 text-center max-w-xs leading-relaxed">
            Secure sign-in with OTP on new devices — India mobile verification.
          </p>
        </div>
        <div className="py-6 flex items-center justify-center gap-2 relative z-10">
          <span className="text-blue-200/50 text-xs">Built by Progressino © 2025</span>
        </div>
      </div>

      <div className="flex-1 flex flex-col bg-gray-50">
        <div className="lg:hidden flex items-center gap-3 px-6 py-5 bg-[#002B5B]">
          <div className="bg-white rounded-lg px-3 py-1.5">
            <img src="/logo.png" alt="Progressino" className="h-7 w-auto" />
          </div>
          <span className="text-white font-semibold text-sm">Yash Gallery ERP</span>
        </div>

        <div className="flex-1 flex items-center justify-center p-8">
          <div className="w-full max-w-sm">
            <div className="mb-8">
              {serverUnreachable && step === 'credentials' && (
                <p className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mb-4">
                  Server was slow to respond. Sign in again — your session may still be valid after login.
                </p>
              )}
              <h2 className="text-2xl font-bold text-gray-900">
                {step === 'otp' ? 'Verify your phone' : 'Welcome back'}
              </h2>
              <p className="text-gray-500 text-sm mt-1">
                {step === 'otp'
                  ? `Enter the 6-digit OTP sent to ${maskedPhone || 'your mobile'}`
                  : 'Sign in to your workspace'}
              </p>
            </div>

            {step === 'credentials' ? (
              <form onSubmit={submitCredentials} className="space-y-5">
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
                    className="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[#002B5B] shadow-sm"
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
                      className="w-full border border-gray-200 rounded-xl px-4 py-3 pr-11 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[#002B5B] shadow-sm"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPwd(v => !v)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400"
                      tabIndex={-1}
                    >
                      {showPwd ? 'Hide' : 'Show'}
                    </button>
                  </div>
                </div>
                {error && <ErrorBox message={error} />}
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-[#002B5B] hover:bg-[#003875] disabled:opacity-50 shadow-md"
                >
                  {loading ? loadStep || 'Signing in…' : 'Sign In'}
                </button>
              </form>
            ) : (
              <form onSubmit={submitOtp} className="space-y-5">
                <div>
                  <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1.5">
                    OTP
                  </label>
                  <input
                    type="text"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    maxLength={6}
                    value={otp}
                    onChange={e => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                    required
                    autoFocus
                    placeholder="6-digit code"
                    className="w-full border border-gray-200 rounded-xl px-4 py-3 text-lg tracking-widest text-center bg-white focus:outline-none focus:ring-2 focus:ring-[#002B5B] shadow-sm"
                  />
                </div>
                <label className="flex items-start gap-2 text-sm text-gray-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={trustDevice}
                    onChange={e => setTrustDevice(e.target.checked)}
                    className="mt-1"
                  />
                  <span>Trust this device for 90 days (skip OTP on this browser)</span>
                </label>
                {error && <ErrorBox message={error} />}
                <button
                  type="submit"
                  disabled={loading || otp.length < 4}
                  className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-[#002B5B] hover:bg-[#003875] disabled:opacity-50 shadow-md"
                >
                  {loading ? loadStep || 'Verifying…' : 'Verify & Sign In'}
                </button>
                <div className="flex justify-between text-xs">
                  <button type="button" onClick={backToCredentials} className="text-gray-500 hover:text-gray-800">
                    ← Back
                  </button>
                  <button type="button" onClick={resendOtp} disabled={loading} className="text-[#002B5B] font-medium">
                    Resend OTP
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 bg-red-50 border border-red-100 rounded-xl px-4 py-3">
      <span className="text-red-500 text-xs">⚠</span>
      <p className="text-red-600 text-xs font-medium">{message}</p>
    </div>
  )
}
