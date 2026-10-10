import { useEffect, useRef, useState } from 'react'

/**
 * The raw log, auto-following the tail.
 *
 * It follows only while the user is already at the bottom. Scrolling up to read
 * something and being yanked back down is the single most annoying thing a live
 * log can do, so position is respected rather than overridden.
 */
export function LogView({ text, running }: { text: string; running: boolean }) {
  const ref = useRef<HTMLPreElement>(null)
  const [pinned, setPinned] = useState(true)

  useEffect(() => {
    const node = ref.current
    if (!node || !pinned) return
    node.scrollTop = node.scrollHeight
  }, [text, pinned])

  const onScroll = () => {
    const node = ref.current
    if (!node) return
    const atBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 24
    setPinned(atBottom)
  }

  const lines = text ? text.split('\n').length : 0

  return (
    <div className="card">
      <h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ display: 'inline-flex', gap: 5, marginRight: 2 }} aria-hidden="true">
            <span style={{ width: 9, height: 9, borderRadius: '50%', background: '#ef4444', opacity: 0.8 }} />
            <span style={{ width: 9, height: 9, borderRadius: '50%', background: '#f59e0b', opacity: 0.8 }} />
            <span style={{ width: 9, height: 9, borderRadius: '50%', background: '#10b981', opacity: 0.8 }} />
          </div>
          <span>Execution Log</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {running && (
            <span className="badge ok" style={{ fontSize: 10.5, padding: '2px 7px' }}>
              <span className="spinner" style={{ width: 8, height: 8 }} />
              LIVE
            </span>
          )}
          <span className="faint mono" style={{ textTransform: 'none', letterSpacing: 0, fontSize: 11.5 }}>
            {lines} lines
          </span>
        </div>
      </h2>
      {!pinned && (
        <button
          className="btn ghost small"
          style={{ marginBottom: 10 }}
          onClick={() => {
            setPinned(true)
            const node = ref.current
            if (node) node.scrollTop = node.scrollHeight
          }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="6 9 12 15 18 9" />
          </svg>
          <span>Jump to latest output</span>
        </button>
      )}
      <pre className="log" ref={ref} onScroll={onScroll}>
        {text || 'Nothing logged yet. Run a step and its output appears here.'}
      </pre>
    </div>
  )
}
