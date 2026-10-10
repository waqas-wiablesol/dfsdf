import { type ActivityItem, type Progress } from '../api'
import { formatClock } from '../useRunState'

/**
 * The live view of a running step.
 *
 * Three things have to be visible at once, because each answers a different
 * question: the bar says how far, the clock says how long it has been going, and
 * the feed says what it has actually done. A long Pro call reports nothing for
 * minutes, and during that window the clock and the feed are the only evidence the
 * run is alive.
 */

function RenderMark({ level }: { level: ActivityItem['level'] }) {
  switch (level) {
    case 'ok':
      return (
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      )
    case 'phase':
      return (
        <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="6 3 20 12 6 21 6 3" />
        </svg>
      )
    case 'error':
      return (
        <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      )
    case 'warn':
      return <span>!</span>
    case 'task':
      return <span style={{ fontSize: 14, lineHeight: 1 }}>•</span>
    default:
      return <span>·</span>
  }
}

export function ProgressPanel({
  running,
  action,
  progress,
  elapsed,
  activity,
  activityTotal,
}: {
  running: boolean
  action: string
  progress: Progress | null
  elapsed: number
  activity: ActivityItem[]
  activityTotal: number
}) {
  if (!running) return null

  const pct = progress?.pct ?? null
  const detail = progress?.detail ?? ''
  const label = action || 'Working...'

  return (
    <div className="card">
      <div className="progress-head">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
          <span className="spinner" style={{ width: 12, height: 12, color: 'var(--brand-teal)' }} />
          <span className="now">{label}</span>
        </div>
        <span className="clock" title="Elapsed runtime">{formatClock(elapsed)}</span>
      </div>

      {/* No number yet gets a dimmed sweep rather than a bar frozen at 0%, which
          reads as stuck when it is really just waiting on the first report. */}
      {pct === null ? (
        <div className="bar indeterminate">
          <i />
        </div>
      ) : (
        <div className="bar">
          <i style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
        </div>
      )}

      <div className="progress-meta">
        <span style={{ fontWeight: 650, color: 'var(--text-primary)' }}>
          {pct === null ? 'Initializing stage...' : `${Math.round(pct)}%`}
        </span>
        {detail && <span>·</span>}
        {detail && <span className="faint">{detail}</span>}
      </div>

      {activity.length > 0 && (
        <div className="feed">
          {activityTotal > activity.length && (
            <div className="small faint" style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between' }}>
              <span>Recent activity</span>
              <span>Showing last {activity.length} of {activityTotal}</span>
            </div>
          )}
          {activity.map((item, index) => (
            <div className={`row lvl-${item.level}`} key={`${item.at}-${index}`}>
              <span className="mark">
                <RenderMark level={item.level} />
              </span>
              <span>{item.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
