import type { StepInfo, StepKey } from '../api'

/**
 * The steps, in order, with their real state.
 *
 * The state comes from files on disk (`jobs.step_state`), not from remembered
 * clicks, so it stays correct after a reload or a CLI run.
 *
 * A step is locked until its predecessor has run, and a locked step is genuinely
 * **not clickable** -- `disabled`, so the browser refuses the click rather than the
 * handler ignoring it. An earlier version left it selectable so its description
 * could be read before committing to the earlier step; that read as "this is
 * clickable", which is the thing the lock exists to prevent. The description is
 * therefore shown inline on the locked row instead, so nothing is lost by making it
 * inert.
 *
 * The server refuses a locked step as well. This is the visible half of that rule,
 * not the enforcement.
 */
export function StepRail({
  steps,
  selected,
  onSelect,
  busy,
  onReset,
}: {
  steps: StepInfo[]
  selected: StepKey
  onSelect: (key: StepKey) => void
  busy: boolean
  onReset?: () => void
}) {
  return (
    <div className="card rail">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>Steps</h2>
        {steps.some((s) => s.done) && onReset && !busy && (
          <button
            className="btn ghost small"
            onClick={onReset}
            title="Reset steps sequence to start fresh with a new video"
            style={{ fontSize: 11, padding: '3px 8px' }}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.19" />
            </svg>
            <span>Reset Steps</span>
          </button>
        )}
      </div>
      {steps.map((step) => {
        const cls = [
          'step',
          step.done ? 'done' : '',
          step.is_next ? 'next' : '',
          step.is_running ? 'running' : '',
          step.locked ? 'locked' : '',
        ]
          .filter(Boolean)
          .join(' ')

        const tag = step.is_running
          ? 'running'
          : step.locked
            ? 'locked'
            : step.done
              ? 'done'
              : step.is_next
                ? 'ready'
                : ''

        return (
          <button
            key={step.key}
            className={cls}
            aria-current={selected === step.key}
            onClick={() => onSelect(step.key)}
            disabled={step.locked}
            aria-disabled={step.locked}
            title={
              step.locked
                ? `${step.label} is locked. ${step.locked_reason}`
                : `Select ${step.label}`
            }
          >
            <span className="dot">
              {step.is_running ? (
                <span className="spinner" style={{ width: 12, height: 12 }} />
              ) : step.locked ? (
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                  <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                </svg>
              ) : step.done ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              ) : (
                <span style={{ fontSize: 11, fontWeight: 700 }}>
                  {step.label.match(/^\d+/)?.[0] ?? ''}
                </span>
              )}
            </span>
            <span className="name">{step.label}</span>
            {tag && <span className="tag">{tag}</span>}
            <span className="hint">
              {step.locked && (
                <span className="hint-reason">{step.locked_reason} </span>
              )}
              {step.help}
            </span>
          </button>
        )
      })}
      {busy && (
        <div className="notice info small" style={{ marginTop: 12 }}>
          <span className="spinner" style={{ width: 13, height: 13 }} />
          <span>
            <b>Step running.</b> Pipeline steps run sequentially so progress, token telemetry, and log output remain precise.
          </span>
        </div>
      )}

      {!busy && (
        <div className="rail-note">
          {steps.every((s) => s.done)
            ? 'All steps completed. Click Reset Steps to start a new video run.'
            : 'Steps run sequentially: each step stays disabled until the step before it is completed.'}
        </div>
      )}
    </div>
  )
}
