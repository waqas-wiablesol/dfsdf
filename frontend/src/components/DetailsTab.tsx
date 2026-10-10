import { useCallback, useEffect, useState } from 'react'
import { api, type DetailsResponse, type Row, type Tokens, type VideoDetail } from '../api'
import { Shot, type ShotView } from './Shot'

/**
 * Per-video detail, and the screenshots at full width.
 *
 * The numbers here are the ones that are otherwise only visible by scrolling
 * through a log: how the video was sliced, how many actions each clip yielded,
 * what it cost in tokens, and how long it took. Two of them need care:
 *
 *  - **Tokens** are recovered from the client's cache for runs that predate usage
 *    being recorded. Where a component genuinely cannot be recovered it is shown
 *    as `not recorded` and the total says it is a lower bound, because a total that
 *    silently omits a source reads as complete.
 *  - **Elapsed** time was not stored before this change and cannot be derived
 *    after the fact, so older runs say so instead of showing 0:00.
 */

function n(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—'
  return value.toLocaleString()
}

function money(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—'
  return value < 0.01 ? `$${value.toFixed(5)}` : `$${value.toFixed(4)}`
}

function secs(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—'
  if (value < 60) return `${value.toFixed(1)}s`
  const m = Math.floor(value / 60)
  return `${m}m ${Math.round(value % 60)}s`
}

function tokensKnown(t: Tokens | undefined): boolean {
  return Boolean(t) && (t!.input !== null || t!.output !== null)
}

export function DetailsTab({ work }: { work: string }) {
  const [data, setData] = useState<DetailsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<ShotView>('full')
  const [shots, setShots] = useState<Row[]>([])
  const [shotsLoading, setShotsLoading] = useState(true)
  // Which videos are folded away, keyed by stem. Absent means expanded, so the
  // initial state is the whole page and folding is an explicit act -- a view that
  // started folded would hide the screenshots that are the reason to open this tab.
  const [folded, setFolded] = useState<Record<string, boolean>>({})

  const load = useCallback(async () => {
    setLoading(true)
    try {
      setData(await api.details(work))
      setError(null)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setLoading(false)
    }
  }, [work])

  // The screenshots come from the rows endpoint, which already knows how to build
  // them. Fetched separately so a slow frame build cannot hold up the numbers.
  useEffect(() => {
    let cancelled = false
    setShotsLoading(true)
    api
      .rows(work)
      .then((r) => {
        if (!cancelled) setShots(r.rows)
      })
      .catch(() => {
        if (!cancelled) setShots([])
      })
      .finally(() => {
        if (!cancelled) setShotsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [work])

  useEffect(() => {
    void load()
  }, [load])

  if (loading) {
    return (
      <div className="body full">
        <div className="col">
          <div className="card">
            <div className="skeleton" style={{ width: '30%', marginBottom: 14 }} />
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <div className="skeleton" key={i} style={{ height: 22, marginBottom: 7 }} />
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="body full">
        <div className="col">
          <div className="notice error">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            <span>
              Could not load the details: {error}
              <div style={{ marginTop: 12 }}>
                <button className="btn" onClick={() => void load()}>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.19" />
                  </svg>
                  Try again
                </button>
              </div>
            </span>
          </div>
        </div>
      </div>
    )
  }

  if (!data || data.video_count === 0) {
    return (
      <div className="body full">
        <div className="col">
          <div className="card">
            <h2>Nothing to show yet</h2>
            <p className="muted">
              Details appear once a video has been through step 2. Run the steps in
              order, then come back.
            </p>
            <button className="btn" onClick={() => void load()}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.19" />
              </svg>
              Reload
            </button>
          </div>
        </div>
      </div>
    )
  }

  const t = data.totals

  return (
    <div className="body full">
      <div className="col">
        <div className="card">
          <h2>Pipeline Totals</h2>
          <div className="stats">
            <Stat label="Videos" value={String(data.video_count)} />
            <Stat label="Clips" value={String(t.clips)} />
            <Stat label="Actions" value={String(t.actions)} />
            <Stat label="Grounded rows" value={String(t.grounded)} />
            <Stat
              label="Tokens in"
              value={n(t.tokens.input)}
              note={t.complete ? undefined : 'lower bound'}
            />
            <Stat
              label="Tokens out"
              value={n(t.tokens.output)}
              note={
                t.tokens.thinking
                  ? `${n(t.tokens.thinking)} thinking`
                  : undefined
              }
            />
            <Stat label="Cost" value={money(t.usd)} note={t.complete ? undefined : 'lower bound'} />
          </div>


          {t.truncated.length > 0 && (
            <div className="notice warn" style={{ marginTop: 12 }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <span>
                <b>
                  {t.truncated.length} video
                  {t.truncated.length === 1 ? ' was' : 's were'} truncated.
                </b>{' '}
                The dataset built from this run does not cover the full video duration.
                <div className="mono small" style={{ marginTop: 6 }}>
                  {t.truncated.map((x) => (
                    <div key={x.stem}>
                      {x.stem}: {x.processed ?? '?'} of {x.planned ?? '?'} clip(s),{' '}
                      {x.covers_pct !== null ? `${x.covers_pct}%` : 'part'} covered
                    </div>
                  ))}
                </div>
              </span>
            </div>
          )}
          {t.limited.length > 0 && (
            <div className="notice warn" style={{ marginTop: 12 }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <span>
                <b>
                  Grounding was capped for {t.limited.length} video
                  {t.limited.length === 1 ? '' : 's'}.
                </b>{' '}
                Some available actions were omitted.
                <div className="mono small" style={{ marginTop: 6 }}>
                  {t.limited.map((x) => (
                    <div key={x.stem}>
                      {x.stem}: {x.attempted ?? '?'} of {x.available ?? '?'} attempted
                    </div>
                  ))}
                </div>
              </span>
            </div>
          )}
        </div>

        {/* The screenshots are grouped by the video they came from and rendered
            inside that video's card, so folding a video away takes its frames with
            it. */}
        {(() => {
          const byVideo: Record<string, Row[]> = {}
          for (const row of shots) {
            const key = row.source_video ?? ''
            const list = byVideo[key]
            if (list) list.push(row)
            else byVideo[key] = [row]
          }
          const allFolded =
            data.videos.length > 0 &&
            data.videos.every((v) => folded[v.stem])

          return (
            <>
              <div className="video-bar">
                <div className="tabs">
                  <button
                    aria-selected={view === 'full'}
                    onClick={() => setView('full')}
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                    <span>Full frame</span>
                  </button>
                  <button
                    aria-selected={view === 'zoom'}
                    onClick={() => setView('zoom')}
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="11" cy="11" r="8" />
                      <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      <line x1="11" y1="8" x2="11" y2="14" />
                      <line x1="8" y1="11" x2="14" y2="11" />
                    </svg>
                    <span>Zoom on target</span>
                  </button>
                </div>
                <span className="pill small" style={{ marginLeft: 4 }}>
                  {shots.length} screenshot{shots.length === 1 ? '' : 's'} · {data.videos.length} video{data.videos.length === 1 ? '' : 's'}
                </span>
                <span style={{ flex: 1 }} />
                <button
                  className="btn ghost small"
                  onClick={() => {
                    if (allFolded) {
                      setFolded({})
                      return
                    }
                    const next: Record<string, boolean> = {}
                    for (const v of data.videos) next[v.stem] = true
                    setFolded(next)
                  }}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points={allFolded ? "7 13 12 18 17 13" : "17 11 12 6 7 11"} />
                    <polyline points={allFolded ? "7 6 12 11 17 6" : "17 18 12 13 7 18"} />
                  </svg>
                  <span>{allFolded ? 'Expand all' : 'Collapse all'}</span>
                </button>
              </div>

              {data.videos.map((video) => (
                <VideoCard
                  key={video.stem}
                  video={video}
                  shots={byVideo[video.video ?? ''] ?? []}
                  shotsLoading={shotsLoading}
                  view={view}
                  folded={Boolean(folded[video.stem])}
                  onToggle={() =>
                    setFolded((prev) => ({
                      ...prev,
                      [video.stem]: !prev[video.stem],
                    }))
                  }
                />
              ))}
            </>
          )
        })()}
      </div>
    </div>
  )
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  )
}

function VideoCard({
  video,
  shots,
  shotsLoading,
  view,
  folded,
  onToggle,
}: {
  video: VideoDetail
  shots: Row[]
  shotsLoading: boolean
  view: ShotView
  folded: boolean
  onToggle: () => void
}) {
  const a = video.actions
  const tk = video.tokens

  // Everything below the header goes inside this, so folding one video cannot
  // leave any of its parts behind. The counts stay in the header while folded --
  // a folded video should still say what it is hiding.
  return (
    <div className={`card video-card${folded ? ' folded' : ''}`}>
      <h2 className="video-head-wrap">
        <button
          className="video-head"
          aria-expanded={!folded}
          onClick={onToggle}
          title={
            folded
              ? 'Show this video, its numbers and its screenshots'
              : 'Fold this video away, including its screenshots'
          }
        >
          <span className="chev" aria-hidden="true" style={{ transform: folded ? 'rotate(-90deg)' : 'rotate(0deg)', display: 'inline-flex', transition: 'transform 0.2s ease' }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </span>
          <span className="video-title">
            <span className="video-name">
              {video.screening?.title || video.video}
            </span>
            <span className="mono small faint video-stem">{video.stem}</span>
          </span>
          <span className="video-meta faint small">
            <span className="pill small">{a.total} action{a.total === 1 ? '' : 's'}</span>
            <span className="pill small" style={{ color: 'var(--brand-teal)' }}>{a.grounded} grounded</span>
            <span className="pill small">{shots.length} shot{shots.length === 1 ? '' : 's'}</span>
            {folded && <span className="video-folded-note">folded</span>}
          </span>
        </button>
      </h2>

      {folded ? null : (
        <>
      <div className="stats">
        <Stat
          label="Duration"
          value={video.duration_s !== null ? secs(video.duration_s) : '—'}
        />
        <Stat
          label="Resolution"
          value={
            video.width && video.height
              ? `${video.width}×${video.height}`
              : '—'
          }
          note={video.fps ? `${video.fps} fps` : undefined}
        />
        <Stat label="Clips" value={String(video.slicing.clip_count)} />
        <Stat
          label="Covered"
          value={secs(video.slicing.total_clip_seconds)}
          note={
            video.slicing.covers_pct !== null
              ? `${video.slicing.covers_pct}% of the video`
              : undefined
          }
        />
        <Stat label="Tasks" value={String(a.tasks)} />
        <Stat label="Actions" value={String(a.total)} />
        <Stat
          label="Groundable"
          value={String(a.groundable)}
          note={`${a.grounded} grounded`}
        />
        <Stat
          label="Yield"
          value={a.yield_pct !== null ? `${a.yield_pct}%` : '—'}
          note={a.dropped ? `${a.dropped} dropped` : undefined}
        />
      </div>

      {/* The single most misleading thing this view could hide: a run capped at a
          few clips produces a record that otherwise reads like the whole video.
          `truncated` is null for runs that predate the flag, so that case says the
          coverage figure and nothing more rather than guessing a verdict. */}
      {video.slicing.truncated === true && (
        <div className="notice warn" style={{ marginTop: 12 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>
            <b>Truncated.</b> Only{' '}
            {video.slicing.processed_clips ?? video.slicing.clip_count} of{' '}
            {video.slicing.planned_clips ?? '?'} clip(s) were processed, covering{' '}
            {video.slicing.covers_pct !== null
              ? `${video.slicing.covers_pct}%`
              : 'part'}{' '}
            of the video. Any row below comes from those minutes only.
          </span>
        </div>
      )}
      {video.slicing.truncated === null &&
        video.slicing.covers_pct !== null &&
        video.slicing.covers_pct < 99 && (
          <div className="notice warn" style={{ marginTop: 12 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>
              The recorded clips cover only{' '}
              <b>{video.slicing.covers_pct}%</b> of this video. This run predates
              the truncation flag, so whether that was deliberate is not recorded.
            </span>
          </div>
        )}

      {/* A capped grounding run and a video with few actions produce the same
          `8/8 grounded`, so the cap is stated rather than left to be inferred. */}
      {video.actions.limited_to !== null && (
        <div className="notice warn" style={{ marginTop: 12 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>
            <b>Grounding was capped.</b> Only{' '}
            {video.actions.limited_to} of{' '}
            {video.actions.available ?? '?'} groundable action(s) were attempted, so
            the rest are absent from the dataset.
          </span>
        </div>
      )}

      {/* ---- slicing ---------------------------------------------------- */}
      <h3 style={{ marginTop: 16 }}>Slicing</h3>
      <div className="table-wrap" style={{ maxHeight: 'none' }}>
        <table className="data">
          <thead>
            <tr>
              <th>clip</th>
              <th>range</th>
              <th>length</th>
              <th>shots</th>
              <th>tasks</th>
              <th>actions</th>
              <th>reply</th>
              <th>tokens in</th>
              <th>tokens out</th>
              <th>cost</th>
              <th>elapsed</th>
            </tr>
          </thead>
          <tbody>
            {video.slicing.clips.map((clip) => (
              <tr key={clip.index}>
                <td className="mono">
                  {clip.index}
                  {clip.cached && <span className="faint"> (cache)</span>}
                </td>
                <td className="num">
                  {Math.round(clip.start_s)}–{Math.round(clip.end_s)}s
                </td>
                <td className="num">{clip.seconds.toFixed(0)}s</td>
                <td className="num">{clip.shots}</td>
                <td className="num">{n(clip.tasks)}</td>
                <td className="num">{n(clip.actions)}</td>
                <td className="num">
                  {clip.reply_chars ? `${(clip.reply_chars / 1000).toFixed(1)}k` : '—'}
                </td>
                <td className="num">{tokensKnown(clip.tokens) ? n(clip.tokens.input) : '—'}</td>
                <td className="num">{tokensKnown(clip.tokens) ? n(clip.tokens.output) : '—'}</td>
                <td className="num">{money(clip.cost_usd)}</td>
                <td className="num">{clip.elapsed_s !== null ? secs(clip.elapsed_s) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small faint" style={{ marginBottom: 0 }}>
        Clips are cut at a {video.clip_seconds}s boundary and downscaled to 720p for
        stage 1 only. Grounding reads the original {video.width}×{video.height}{' '}
        video, so the coordinates are in full-resolution pixels.
      </p>

      {/* ---- actions ---------------------------------------------------- */}
      <h3 style={{ marginTop: 16 }}>Actions extracted</h3>
      <div className="action-types">
        {Object.entries(a.by_type).map(([type, count]) => (
          <span className="pill" key={type}>
            {type} <b>{count}</b>
          </span>
        ))}
      </div>
      <p className="small faint">
        {n(a.groundable)} of {n(a.total)} actions carry a coordinate, so{' '}
        {n(a.total - a.groundable)} need none (a `finish` action has nowhere to
        click). {n(a.grounded)} grounded
        {a.avg_attempts ? `, averaging ${a.avg_attempts} frame(s) per action` : ''}.
        Cross-check against stage 1 rejected {n(a.dropped)}.
      </p>

      {/* ---- tokens ----------------------------------------------------- */}
      <h3 style={{ marginTop: 16 }}>Tokens and time</h3>
      <div className="table-wrap" style={{ maxHeight: 'none' }}>
        <table className="data">
          <thead>
            <tr>
              <th>stage</th>
              <th>model</th>
              <th>calls</th>
              <th>tokens in</th>
              <th>tokens out</th>
              <th>thinking</th>
              <th>cost</th>
              <th>elapsed</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Stage 1 — trajectory</td>
              <td className="mono small">{video.trajectory_model}</td>
              <td className="num">{video.slicing.clip_count}</td>
              <td className="num">{tokensKnown(tk.trajectory) ? n(tk.trajectory.input) : 'not recorded'}</td>
              <td className="num">{tokensKnown(tk.trajectory) ? n(tk.trajectory.output) : '—'}</td>
              <td className="num">{tokensKnown(tk.trajectory) ? n(tk.trajectory.thinking) : '—'}</td>
              <td className="num">{money(tk.trajectory_usd)}</td>
              <td className="num">{video.elapsed.trajectory_s !== null ? secs(video.elapsed.trajectory_s) : 'not recorded'}</td>
            </tr>
            <tr>
              <td>Stage 2 — grounding</td>
              <td className="mono small">{video.grounding_model}</td>
              <td className="num">{a.grounded + a.dropped}</td>
              <td className="num">{tokensKnown(tk.grounding) ? n(tk.grounding.input) : 'not recorded'}</td>
              <td className="num">{tokensKnown(tk.grounding) ? n(tk.grounding.output) : '—'}</td>
              <td className="num">{tokensKnown(tk.grounding) ? n(tk.grounding.thinking) : '—'}</td>
              <td className="num">{money(tk.grounding_usd)}</td>
              <td className="num">{video.elapsed.grounding_s !== null ? secs(video.elapsed.grounding_s) : 'not recorded'}</td>
            </tr>
            <tr>
              <td>
                <b>Total</b>
              </td>
              <td />
              <td className="num">{video.slicing.clip_count + a.grounded + a.dropped}</td>
              <td className="num">
                <b>{n(tk.total.input)}</b>
              </td>
              <td className="num">
                <b>{n(tk.total.output)}</b>
              </td>
              <td className="num">{n(tk.total.thinking)}</td>
              <td className="num">
                <b>{money(tk.total_usd)}</b>
              </td>
              <td className="num">
                {video.elapsed.recorded
                  ? secs((video.elapsed.trajectory_s ?? 0) + (video.elapsed.grounding_s ?? 0))
                  : 'not recorded'}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      {tk.unknown.length > 0 && (
        <p className="small warn-text">
          {tk.unknown.join(' and ')} usage is not recorded in this run's files and
          could not be recovered from the cache, so the total is a lower bound.
        </p>
      )}

      {/* ---- screenshots ------------------------------------------------ */}
      <h3 style={{ marginTop: 16 }}>
        Screenshots
        <span
          className="faint small"
          style={{ marginLeft: 8, fontWeight: 400 }}
        >
          {shots.length} from this video
        </span>
      </h3>

      {shotsLoading ? (
        <div>
          {[0, 1].map((i) => (
            <div
              key={i}
              className="skeleton"
              style={{ height: 180, marginBottom: 10 }}
            />
          ))}
        </div>
      ) : shots.length === 0 ? (
        <p className="small faint" style={{ marginBottom: 0 }}>
          No screenshot from this video. A row only exists when its action grounded
          AND its frame is on disk, so this means either nothing here grounded or
          the frames are missing -- run step 5 to recover them, then step 4 to
          include the rows.
        </p>
      ) : (
        <div className={view === 'full' ? 'shots' : 'crops'}>
          {shots.map((row, index) => (
            <Shot key={row.id ?? index} row={row} view={view} />
          ))}
        </div>
      )}
        </>
      )}
    </div>
  )
}
