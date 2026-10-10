import { useEffect, useState } from 'react'
import { api, type Options, type StepKey } from '../api'
import type { RunApi } from '../useRunState'
import { LogView } from './LogView'
import { PlaylistQueueCard } from './PlaylistQueueCard'
import { ProgressPanel } from './ProgressPanel'
import { StepRail } from './StepRail'
import { UrlDetectionCard } from './UrlDetectionCard'

/**
 * The run view: pick a step, run it, watch it.
 *
 * Each step is its own button and nothing chains -- the step runs and stops, so an
 * intermediate result can be inspected before the next paid call. That is the
 * whole point of the layout.
 */

/** Which inputs a step actually reads. Showing all every time invites typos. */
const FIELDS: Record<StepKey, Array<'url' | 'source' | 'thinking'>> = {
  screen: ['url'],
  download: ['url', 'source'],
  trajectory: ['thinking'],
  grounding: [],
  dataset: [],
  review: [],
  autoflow: ['url', 'source', 'thinking'],
  playlist: [],
}

/** Steps whose output lives in the Results tab, so it can be linked from here. */
const SHOWS_IN_RESULTS: StepKey[] = ['dataset', 'review']

export function RunTab({ work, run, onChangeWork, onOpenResults }: {
  work: string
  run: RunApi
  onChangeWork: (value: string) => void
  onOpenResults: () => void
}) {
  const { state, starting, error, start, clear } = run
  const [selected, setSelected] = useState<StepKey>('screen')
  const [runMode, setRunMode] = useState<'step' | 'playlist'>(() => {
    return (run.state?.step as string) === 'playlist' ? 'playlist' : 'step'
  })
  const [url, setUrl] = useState('')
  const [playlistUrl, setPlaylistUrl] = useState('')
  const [source, setSource] = useState('')
  const [thinking, setThinking] = useState('medium')
  const [options, setOptions] = useState<Options | null>(null)

  useEffect(() => {
    if ((state?.step as string) === 'playlist') {
      setRunMode('playlist')
    }
  }, [state?.step])

  // Fetched rather than hard-coded: the levels and the notes beside them come from
  // `config`, so there is one copy of the list and the numbers quoted are the
  // measurements, not a second-hand paraphrase.
  useEffect(() => {
    api
      .options()
      .then((opts) => {
        setOptions(opts)
        if (opts.default_thinking) {
          setThinking(opts.default_thinking)
        }
      })
      .catch(() => setOptions(null))
  }, [])

  const running = state?.running ?? false
  const busy = running || starting

  // The selected step is NEVER moved automatically. It used to follow the
  // pipeline (jump to whatever was next once a step finished), which read as the
  // UI advancing on its own -- the exact thing that must not happen. A step runs
  // only when its own button is pressed, so leaving the selection alone keeps the
  // reading honest: after a run, the step you were looking at is still there.

  const onRun = () => {
    const payload: Record<string, unknown> = {}
    if (FIELDS[selected].includes('url')) payload.url = url
    if (FIELDS[selected].includes('source')) payload.source = source
    if (FIELDS[selected].includes('thinking')) payload.thinking = thinking
    void start(selected, payload)
  }

  const onAutoflow = (targetUrl?: string) => {
    const rawUrl = (targetUrl ?? url).trim()
    const payload: Record<string, unknown> = {}
    if (rawUrl) payload.url = rawUrl
    if (source.trim()) payload.source = source.trim()
    if (thinking) payload.thinking = thinking
    void start('autoflow', payload)
  }

  const stepInfo = state?.steps.find((s) => s.key === selected)
  // A locked step cannot run, and the reason is shown rather than leaving a dead
  // button. The server refuses it as well, so this is the explanation and not the
  // enforcement.
  const locked = Boolean(stepInfo?.locked)
  const needsInput =
    (selected === 'screen' || selected === 'download') &&
    !url.trim() &&
    !source.trim()
  const canRun = !busy && !locked && !needsInput
  const canAutoflow = !busy && (Boolean(url.trim()) || Boolean(source.trim()))

  // No state yet. Either it is still arriving (skeleton) or it failed -- and a
  // skeleton that never resolves is the worst outcome, because it looks like work
  // is happening when nothing is. A failure says so and offers a retry.
  if (!state) {
    if (error) {
      return (
        <div className="body">
          <div className="col">
            <div className="notice error">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              <span>
                <b>Could not reach the API.</b> Nothing is currently executing.
                <div className="mono small" style={{ marginTop: 6, overflowWrap: 'anywhere', opacity: 0.9 }}>
                  {error}
                </div>
                <div style={{ marginTop: 12 }}>
                  <button className="btn" onClick={() => run.refresh()}>
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.19" />
                    </svg>
                    Retry connection
                  </button>
                </div>
              </span>
            </div>
          </div>
        </div>
      )
    }
    return (
      <div className="body">
        <div className="col">
          <div className="card">
            <h2>Pipeline Steps</h2>
            <div className="faint small" style={{ marginBottom: 12 }}>
              Connecting to the pipeline API...
            </div>
            {[0, 1, 2, 3, 4].map((i) => (
              <div className="skeleton" key={i} style={{ height: 46, marginBottom: 10 }} />
            ))}
          </div>
        </div>
        <div className="col">
          <div className="card">
            <div className="skeleton" style={{ width: '35%', marginBottom: 16 }} />
            <div className="skeleton" style={{ height: 110 }} />
          </div>
        </div>
      </div>
    )
  }

  return (
    <>
      <div style={{ maxWidth: 1720, margin: '0 auto', padding: '16px 24px 0', width: '100%' }}>
        <div className="pipeline-mode-bar">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)' }}>
              Execution Mode:
            </span>
            <div className="pipeline-mode-tabs">
              <button
                className={`pipeline-mode-tab ${runMode === 'step' ? 'active' : ''}`}
                onClick={() => setRunMode('step')}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="8" y1="6" x2="21" y2="6" />
                  <line x1="8" y1="12" x2="21" y2="12" />
                  <line x1="8" y1="18" x2="21" y2="18" />
                  <line x1="3" y1="6" x2="3.01" y2="6" />
                  <line x1="3" y1="12" x2="3.01" y2="12" />
                  <line x1="3" y1="18" x2="3.01" y2="18" />
                </svg>
                <span>Single Video (Step-by-Step)</span>
              </button>
              <button
                className={`pipeline-mode-tab playlist ${runMode === 'playlist' ? 'active' : ''}`}
                onClick={() => setRunMode('playlist')}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z" />
                  <polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02" fill="currentColor" />
                </svg>
                <span>YouTube Playlist Queue (SQLite)</span>
              </button>
            </div>
          </div>

          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {runMode === 'playlist' ? 'Sequential SQLite queue pipeline' : 'Controlled manual step execution'}
          </div>
        </div>
      </div>

      <div className="body">
        <div className="col">
          {runMode === 'step' ? (
            <StepRail
              steps={state.steps}
              selected={selected}
              onSelect={setSelected}
              busy={running}
              onReset={() => void clear()}
            />
          ) : (
            <div className="card">
              <h2>Playlist Queue Stages</h2>
              <div className="faint small" style={{ marginBottom: 16 }}>
                Videos in the SQLite queue cycle automatically through these 4 sequential pipeline stages:
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                  <span className="q-status downloading" style={{ flex: 'none', marginTop: 2 }}>1. Download</span>
                  <span className="small muted">Fetches 1080p MP4 stream into <code className="mono">{work}/source</code>.</span>
                </div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                  <span className="q-status trajectory" style={{ flex: 'none', marginTop: 2 }}>2. Trajectory</span>
                  <span className="small muted">Stage 1 multimodal reasoning extracts structured actions and timestamps.</span>
                </div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                  <span className="q-status grounding" style={{ flex: 'none', marginTop: 2 }}>3. Grounding</span>
                  <span className="small muted">Stage 2 resolves exact UI coordinates on high-res extracted video frames.</span>
                </div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                  <span className="q-status completed" style={{ flex: 'none', marginTop: 2 }}>4. Dataset</span>
                  <span className="small muted">Auto-joins valid grounded actions directly into <code className="mono">unified.parquet</code>.</span>
                </div>
              </div>

              <div style={{ marginTop: 22, paddingTop: 16, borderTop: '1px solid var(--border-medium)' }}>
                <h3 style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: 8 }}>
                  Queue Guarantees
                </h3>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.5, color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <li><b>Sequential:</b> Only 1 video processes at a time to prevent API rate limits.</li>
                  <li><b>Idempotent:</b> Completed videos are permanently preserved and skipped.</li>
                  <li><b>Fault-tolerant:</b> A failed video is marked and the queue safely continues.</li>
                </ul>
              </div>
            </div>
          )}
        </div>

        <div className="col">
          {(error || state.error) && (
            <div className="notice error">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              <span>{state.error ?? error}</span>
            </div>
          )}

          {runMode === 'playlist' ? (
            <PlaylistQueueCard
              work={work}
              busy={busy}
              onOpenResults={onOpenResults}
              initialUrl={playlistUrl}
              onSwitchToStep={(videoUrl) => {
                setUrl(videoUrl)
                setRunMode('step')
              }}
            />
          ) : (
            <div className="card">
              <h2>{running ? 'Pipeline active' : 'Step configuration'}</h2>
              <h3>{stepInfo?.label ?? selected}</h3>
              <p className="muted small" style={{ marginTop: 0 }}>
                {stepInfo?.help}
              </p>

          {FIELDS[selected].includes('url') && (
            <div className="field">
              <label htmlFor="url">YouTube URL</label>
              <input
                id="url"
                value={url}
                placeholder="https://www.youtube.com/watch?v=..."
                onChange={(e) => setUrl(e.target.value)}
                disabled={busy}
              />
              <UrlDetectionCard
                url={url}
                currentMode="step"
                stepName={stepInfo?.label ?? selected}
                onSwitchToPlaylist={(detectedUrl) => {
                  setPlaylistUrl(detectedUrl)
                  setRunMode('playlist')
                }}
                onSelectSingleVideo={(singleUrl) => {
                  setUrl(singleUrl)
                }}
              />
            </div>
          )}


          {FIELDS[selected].includes('source') && (
            <div className="field">
              <label htmlFor="source">or a folder that already holds videos</label>
              <input
                id="source"
                value={source}
                placeholder="C:/videos"
                onChange={(e) => setSource(e.target.value)}
                disabled={busy}
              />
              <span className="note">
                Leave both empty to reuse whatever is already in <code>{work}/source</code>.
              </span>
            </div>
          )}


          {FIELDS[selected].includes('thinking') && (
            <div className="field">
              <label htmlFor="thinking">Thinking effort</label>
              <select
                id="thinking"
                value={thinking}
                onChange={(e) => setThinking(e.target.value)}
                disabled={busy}
              >
                <option value="">
                  Model default (
                  {options?.default_thinking ?? 'medium'})
                </option>
                {(options?.thinking_levels ?? []).map((level) => (
                  <option key={level.value} value={level.value}>
                    {level.value}
                  </option>
                ))}
              </select>
              <span className="note">
                {(thinking && options?.thinking_levels.find((l) => l.value === thinking)?.note) ||
                  options?.thinking_levels.find((l) => l.value === (options?.default_thinking ?? 'medium'))?.note ||
                  'a middle setting. Around 8,000 thinking tokens.'}
              </span>
              {/* Each level is a separate cache entry, so switching invalidates
                  nothing -- but mixed levels in one work tree would make the
                  dataset inconsistent, and that is worth saying. */}
              <span className="note">
                Changing this re-annotates the video: a record made at one level is
                not reused for another, so the existing trajectories for{' '}
                {work} would be replaced.
              </span>
            </div>
          )}


          <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 10, flexWrap: 'wrap' }}>
            <button
              className="btn primary"
              onClick={() => onAutoflow()}
              disabled={!canAutoflow}
              title={
                canAutoflow
                  ? 'Automatically execute Download -> Trajectory -> Grounding -> Dataset -> Review on its own'
                  : 'Enter a URL or local video folder to start full Auto-Flow'
              }
              style={{
                background: 'linear-gradient(135deg, #0ea5e9, #2563eb)',
                borderColor: '#38bdf8',
                color: '#fff',
                boxShadow: '0 2px 10px rgba(14, 165, 233, 0.3)',
              }}
            >
              {running && state.step === 'autoflow' ? (
                <span className="spinner" style={{ width: 14, height: 14 }} />
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
              )}
              <span>{running && state.step === 'autoflow' ? 'Auto-Flow Running...' : 'Start Auto-Flow (Full Pipeline)'}</span>
            </button>

            <button
              className="btn"
              onClick={onRun}
              disabled={!canRun}
              title={`Execute only this individual step: ${stepInfo?.label ?? selected}`}
            >
              {busy && state.step !== 'autoflow' ? (
                <span className="spinner" style={{ width: 14, height: 14 }} />
              ) : (
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="9 18 15 12 9 6" />
                </svg>
              )}
              <span>
                {running && state.step !== 'autoflow'
                  ? `Running ${state.label}`
                  : starting
                    ? 'Starting...'
                    : locked
                      ? 'Step locked'
                      : `Run Step: ${stepInfo?.label ?? selected}`}
              </span>
            </button>

            {(state.result || state.activity.length > 0) && !running && (
              <button className="btn ghost" onClick={() => void clear()}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
                Clear result
              </button>
            )}

            {!running && SHOWS_IN_RESULTS.includes(selected) && (
              <button className="btn" onClick={onOpenResults}>
                <span>Open screenshots</span>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>
            )}
          </div>

          {!canRun && !busy && (
            <div className="note faint" style={{ marginTop: 10 }}>
              {locked
                ? stepInfo?.locked_reason
                : 'Enter a URL or source directory to enable this step.'}
            </div>
          )}

            {locked && (
              <div className="notice warn" style={{ marginTop: 12 }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                  <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                </svg>
                <span>
                  <b>Step locked.</b> Pipeline steps run in order: {stepInfo?.locked_reason}{' '}
                  This prevents costly downstream stages from executing against missing or unverified inputs.
                </span>
              </div>
            )}
          </div>
        )}

        <ProgressPanel
          running={running}
          action={state.action}
          progress={state.progress}
          elapsed={state.elapsed}
          activity={state.activity}
          activityTotal={state.activity_total}
        />

        {!running && state.result && <ResultSummary result={state.result} />}

        <LogView text={state.log} running={running} />

        <div className="card">
          <h2>Work directory</h2>
          <div className="field" style={{ marginBottom: 0 }}>
            <input value={work} onChange={(e) => onChangeWork(e.target.value)} />
            <span className="note">
              Target storage path for clips, intermediate stage JSON records, extracted PNG frames, and final Parquet datasets.
            </span>
          </div>
        </div>
      </div>
    </div>
    </>
  )
}

function ResultSummary({ result }: { result: Record<string, unknown> }) {
  const shown = Object.entries(result).filter(([, v]) => v !== null && v !== '' && v !== undefined)
  if (shown.length === 0) return null

  return (
    <div className="card">
      <h2>Result</h2>
      <dl className="kv">
        {shown.map(([key, value]) => (
          <div key={key} style={{ display: 'contents' }}>
            <dt>{key.replace(/_/g, ' ')}</dt>
            <dd className="mono small">{describe(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function describe(value: unknown): string {
  if (Array.isArray(value)) {
    if (value.length > 3) return `${value.slice(0, 3).join(', ')} and ${value.length - 3} more`
    return value.join(', ')
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
    return entries.map(([k, v]) => `${k}: ${v}`).join('  ·  ')
  }
  return String(value)
}
