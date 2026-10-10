import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type RunState, type StepKey } from './api'

/**
 * Polls the run state.
 *
 * Polling rather than server-sent events on purpose. The browser aborted Gradio's
 * SSE stream on every reconnect, which is exactly why the earlier UI looked dead
 * after a click; a short GET against localhost cannot fail that way.
 *
 * The interval is not fixed: 400 ms while a step is in flight, 2.5 s when idle.
 * A run that reports every 400 ms needs the fast cadence, and an idle tab should
 * not hammer the server for no reason.
 */

const FAST_MS = 400
const IDLE_MS = 2500

export interface RunApi {
  state: RunState | null
  /** True only for the very first fetch, so the UI can show a skeleton. */
  loading: boolean
  /** True between pressing Run and the poll that reports it running. */
  starting: boolean
  error: string | null
  start: (key: StepKey, payload: Record<string, unknown>) => Promise<void>
  clear: () => Promise<void>
  /** Refetch now instead of waiting for the next tick. */
  refresh: () => void
}

export function useRunState(work: string): RunApi {
  const [state, setState] = useState<RunState | null>(null)
  const [loading, setLoading] = useState(true)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Incremented to restart the polling loop immediately: after a work-dir change,
  // after pressing Run, or after any action whose result should not wait.
  const [generation, setGeneration] = useState(0)

  const workRef = useRef(work)
  useEffect(() => {
    workRef.current = work
  }, [work])

  useEffect(() => {
    let stopped = false
    let timer: number | undefined
    let first = true

    const loop = async () => {
      if (stopped) return
      let delay = IDLE_MS
      try {
        const next = await api.state(workRef.current)
        if (stopped) return
        setState(next)
        setError(null)
        delay = next.running ? FAST_MS : IDLE_MS
      } catch (exc) {
        if (stopped) return
        setError(exc instanceof Error ? exc.message : String(exc))
        delay = IDLE_MS
      } finally {
        if (!stopped) {
          if (first) {
            first = false
            setLoading(false)
          }
          timer = window.setTimeout(loop, delay)
        }
      }
    }

    loop()
    return () => {
      stopped = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [generation, work])

  const refresh = useCallback(() => setGeneration((n) => n + 1), [])

  const start = useCallback(
    async (key: StepKey, payload: Record<string, unknown>) => {
      setError(null)
      setStarting(true)
      try {
        const result = await api.run(key, { work, ...payload })
        if (!result.started) setError(result.message)
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : String(exc))
      } finally {
        setStarting(false)
        // Whether it started or was refused, look again at once rather than
        // leaving the user staring at a button that appears to have done nothing.
        setGeneration((n) => n + 1)
      }
    },
    [work],
  )

  const clear = useCallback(async () => {
    try {
      await api.reset(workRef.current)
      setState((prev) =>
        prev ? { ...prev, result: null, error: null, activity: [], action: '' } : prev,
      )
      setGeneration((n) => n + 1)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    }
  }, [])

  return { state, loading, starting, error, start, clear, refresh }
}

/** `92` -> `1:32`. Missing or non-finite input renders as `-`. */
export function formatClock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '-'
  const total = Math.max(0, Math.floor(seconds))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${m}:${String(s).padStart(2, '0')}`
}
