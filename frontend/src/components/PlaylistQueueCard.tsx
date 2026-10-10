import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, type QueueItem } from '../api'
import { UrlDetectionCard } from './UrlDetectionCard'

interface PlaylistQueueCardProps {
  work: string
  busy: boolean
  onOpenResults: () => void
  initialUrl?: string
  onSwitchToStep?: (videoUrl: string) => void
}

function formatDuration(seconds: number | null): string {
  if (seconds == null || seconds <= 0) return '--:--'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function PlaylistQueueCard({
  work,
  busy,
  onOpenResults,
  initialUrl,
  onSwitchToStep,
}: PlaylistQueueCardProps) {
  const [url, setUrl] = useState(initialUrl || '')
  const [enqueuing, setEnqueuing] = useState(false)
  const [inspectTrigger, setInspectTrigger] = useState(0)
  const [notice, setNotice] = useState<{ type: 'ok' | 'error'; message: string } | null>(null)

  useEffect(() => {
    if (initialUrl && initialUrl !== url) {
      setUrl(initialUrl)
    }
  }, [initialUrl])

  const [items, setItems] = useState<QueueItem[]>([])
  const [stats, setStats] = useState<Record<string, number>>({})
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState('')
  const [actionBusy, setActionBusy] = useState(false)

  const fetchQueue = useCallback(async () => {
    try {
      const res = await api.queue(work)
      setItems(res.items || [])
      setStats(res.stats || {})
    } catch {
      // queue endpoint silently handles offline or pending start
    }
  }, [work])

  useEffect(() => {
    setLoading(true)
    fetchQueue().finally(() => setLoading(false))
  }, [fetchQueue])

  // Poll more aggressively when busy or active items exist in queue
  const hasActive = useMemo(() => {
    return items.some((i) => i.status === 'downloading' || i.status === 'trajectory' || i.status === 'grounding')
  }, [items])

  useEffect(() => {
    const intervalMs = busy || hasActive ? 2500 : 8000
    const timer = setInterval(() => {
      void fetchQueue()
    }, intervalMs)
    return () => clearInterval(timer)
  }, [fetchQueue, busy, hasActive])

  const enqueueWithUrl = async (targetUrl: string) => {
    const trimmed = targetUrl.trim()
    if (!trimmed) return

    setEnqueuing(true)
    setNotice(null)
    try {
      const res = await api.enqueue(trimmed, work)
      const data = res.result as { total_extracted?: number; playlist_title?: string } | undefined
      const count = data?.total_extracted ?? 1
      const plTitle = data?.playlist_title ? ` ("${data.playlist_title}")` : ''
      setNotice({
        type: 'ok',
        message: `Successfully enqueued ${count} video(s)${plTitle} into SQLite queue.`,
      })
      setUrl('')
      await fetchQueue()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setNotice({ type: 'error', message: msg })
    } finally {
      setEnqueuing(false)
    }
  }

  const onStartAutoflowQueue = async (targetUrl?: string) => {
    const rawUrl = (targetUrl ?? url).trim()
    setActionBusy(true)
    setNotice(null)
    try {
      const res = await api.autoflowQueue({
        url: rawUrl,
        work,
      })
      if (!res.started) {
        setNotice({ type: 'error', message: res.message || 'Pipeline failed to start.' })
      } else {
        const enq = res.enqueued as { total_extracted?: number; playlist_title?: string } | undefined
        const plMsg = enq?.total_extracted ? ` Enqueued ${enq.total_extracted} video(s).` : ''
        setNotice({ type: 'ok', message: `Playlist Auto-Flow started in the background.${plMsg}` })
        if (rawUrl && rawUrl === url) setUrl('')
      }
      await fetchQueue()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setNotice({ type: 'error', message: msg })
    } finally {
      setActionBusy(false)
    }
  }

  const onRetryFailed = async () => {
    setActionBusy(true)
    try {
      const res = await api.retryQueue(work)
      setNotice({ type: 'ok', message: `Reset ${res.reset_count} failed video(s) back to pending.` })
      await fetchQueue()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setNotice({ type: 'error', message: msg })
    } finally {
      setActionBusy(false)
    }
  }

  const onClear = async (mode: 'all' | 'finished') => {
    if (mode === 'all' && !window.confirm('Clear all videos from the queue?')) return
    setActionBusy(true)
    try {
      const res = await api.clearQueue(work, mode)
      setNotice({ type: 'ok', message: `Removed ${res.cleared_count} item(s) from the queue.` })
      await fetchQueue()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setNotice({ type: 'error', message: msg })
    } finally {
      setActionBusy(false)
    }
  }

  // Filtered rows
  const filteredItems = useMemo(() => {
    if (!filter.trim()) return items
    const q = filter.toLowerCase()
    return items.filter(
      (item) =>
        item.title.toLowerCase().includes(q) ||
        item.video_id.toLowerCase().includes(q) ||
        (item.playlist_title && item.playlist_title.toLowerCase().includes(q))
    )
  }, [items, filter])

  const pendingCount = (stats['pending'] || 0)
  const failedCount = (stats['failed'] || 0)
  const completedCount = (stats['completed'] || 0)
  const activeCount =
    (stats['downloading'] || 0) +
    (stats['trajectory'] || 0) +
    (stats['grounding'] || 0)
  const totalCount = items.length

  const canRun = !busy && !actionBusy && (pendingCount > 0 || failedCount > 0)

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h2 style={{ marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--brand-teal)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z" />
              <polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02" fill="var(--brand-teal)" />
            </svg>
            <span>YouTube Playlist Sequential Ingestion</span>
          </h2>
          <p className="muted small" style={{ marginTop: 0, maxWidth: 740 }}>
            Enqueue an entire YouTube playlist. Each video is parsed instantly, stored in an SQLite queue (<span className="mono">queue.db</span>),
            and processed sequentially through download, trajectory extraction, and grounding. Completed videos are cached and automatically joined
            into the single unified dataset.
          </p>
        </div>

        {completedCount > 0 && (
          <button className="btn" onClick={onOpenResults}>
            <span>View Unified Dataset</span>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
        )}
      </div>

      {notice && (
        <div className={`notice ${notice.type === 'ok' ? 'info' : 'error'}`} style={{ marginTop: 12 }}>
          {notice.type === 'ok' ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
              <polyline points="20 6 9 17 4 12" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
          )}
          <span>{notice.message}</span>
        </div>
      )}

      {/* URL Input Form */}
      <form
        onSubmit={(e) => {
          e.preventDefault()
          setInspectTrigger((t) => t + 1)
        }}
        style={{ marginTop: 14 }}
      >
        <div className="field">
          <label htmlFor="playlist-url">YouTube Playlist or Video URL</label>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <input
              id="playlist-url"
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/playlist?list=PL... or https://youtu.be/..."
              disabled={enqueuing || actionBusy}
              style={{ flex: 1, minWidth: 280 }}
            />
            <button
              type="button"
              className="btn primary"
              onClick={() => void onStartAutoflowQueue(url)}
              disabled={actionBusy || busy || !url.trim()}
              style={{ flex: 'none' }}
              title="Automatically enqueues all videos from the playlist and runs download -> trajectory -> grounding -> dataset sequentially on its own"
            >
              {actionBusy ? (
                <>
                  <span className="spinner" style={{ width: 14, height: 14 }} />
                  <span>Starting Auto-Flow...</span>
                </>
              ) : (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  <span>Start Playlist Auto-Flow</span>
                </>
              )}
            </button>
            <button
              type="button"
              className="btn ghost"
              onClick={() => void enqueueWithUrl(url)}
              disabled={enqueuing || actionBusy || !url.trim()}
              style={{ flex: 'none' }}
              title="Add videos into SQLite queue.db in pending status (does not auto-run)"
            >
              {enqueuing ? (
                <>
                  <span className="spinner" style={{ width: 14, height: 14 }} />
                  <span>Adding to Queue...</span>
                </>
              ) : (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="12" y1="5" x2="12" y2="19" />
                    <line x1="5" y1="12" x2="19" y2="12" />
                  </svg>
                  <span>Add to Queue (Pending)</span>
                </>
              )}
            </button>
            <button
              type="submit"
              className="btn"
              disabled={!url.trim() || actionBusy}
              style={{ flex: 'none' }}
              title="Test retrieving metadata without downloading or queuing"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <span>Test Metadata</span>
            </button>
          </div>
          <span className="note">
            <b>Auto-Flow:</b> Entering a URL inspects metadata in seconds. Click <b>Start Playlist Auto-Flow</b> to enqueue and process all videos sequentially on their own through download, trajectory, grounding, and dataset assembly.
          </span>
          <UrlDetectionCard
            url={url}
            currentMode="playlist"
            inspectTrigger={inspectTrigger}
            onSwitchToStep={onSwitchToStep}
          />
        </div>
      </form>


      {/* Statistics Cards */}
      <div className="q-stats-grid">
        <div className="q-stat-card total">
          <span className="q-stat-num">{totalCount}</span>
          <span className="q-stat-lbl">Total Queued</span>
        </div>
        <div className="q-stat-card pending">
          <span className="q-stat-num">{pendingCount}</span>
          <span className="q-stat-lbl">Pending</span>
        </div>
        <div className="q-stat-card active">
          <span className="q-stat-num">{activeCount}</span>
          <span className="q-stat-lbl">Processing</span>
        </div>
        <div className="q-stat-card completed">
          <span className="q-stat-num">{completedCount}</span>
          <span className="q-stat-lbl">Completed</span>
        </div>
        <div className="q-stat-card failed">
          <span className="q-stat-num">{failedCount}</span>
          <span className="q-stat-lbl">Failed</span>
        </div>
      </div>

      {/* Control Action Toolbar */}
      <div className="q-toolbar">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <button
            className="btn primary"
            onClick={() => void onStartAutoflowQueue()}
            disabled={!canRun}
            title={canRun ? 'Start sequential pipeline over all pending items' : 'No pending items or pipeline is busy'}
          >
            {busy || actionBusy ? (
              <span className="spinner" style={{ width: 14, height: 14 }} />
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
            )}
            <span>{busy ? 'Auto-Flow Running...' : 'Start Playlist Auto-Flow'}</span>
          </button>

          {failedCount > 0 && (
            <button
              className="btn"
              onClick={onRetryFailed}
              disabled={busy || actionBusy}
              title="Reset all failed videos back to pending"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.19" />
              </svg>
              <span>Retry Failed ({failedCount})</span>
            </button>
          )}

          {completedCount > 0 && (
            <button
              className="btn ghost small"
              onClick={() => onClear('finished')}
              disabled={busy || actionBusy}
              title="Remove completed items from queue list"
            >
              <span>Clear Completed</span>
            </button>
          )}

          {totalCount > 0 && (
            <button
              className="btn ghost small"
              onClick={() => onClear('all')}
              disabled={busy || actionBusy}
              title="Empty the entire queue"
            >
              <span>Clear All</span>
            </button>
          )}
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            type="text"
            className="table-filter-input"
            placeholder="Search queue..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ width: 180, padding: '4px 10px', fontSize: 12 }}
          />
          <button
            className="btn ghost small"
            onClick={() => void fetchQueue()}
            disabled={loading}
            title="Refresh queue status"
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className={loading ? 'spinning' : ''}
            >
              <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.19" />
            </svg>
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Queue Table */}
      {totalCount === 0 ? (
        <div style={{ padding: '36px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ opacity: 0.4, marginBottom: 8 }}>
            <rect x="2" y="4" width="20" height="16" rx="2" />
            <path d="M10 4v4M14 4v4M4 8h16" />
          </svg>
          <div style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>
            Queue is empty
          </div>
          <div className="small faint" style={{ maxWidth: 460, margin: '0 auto' }}>
            Paste a YouTube playlist link above and click <b>Enqueue Playlist</b> to populate the SQLite queue. Videos will run sequentially one by one.
          </div>
        </div>
      ) : (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th style={{ width: 44, textAlign: 'center' }}>#</th>
                <th style={{ width: 110 }}>Status</th>
                <th>Title & URL</th>
                <th style={{ width: 80, textAlign: 'right' }}>Duration</th>
                <th style={{ width: 90, textAlign: 'center' }}>Actions</th>
                <th style={{ width: 90, textAlign: 'center' }}>Grounded</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {filteredItems.map((item) => (
                <tr key={item.video_id}>
                  <td style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 12, opacity: 0.7 }}>
                    {item.index_in_list}
                  </td>
                  <td>
                    <span className={`q-status ${item.status}`}>
                      {item.status}
                    </span>
                  </td>
                  <td>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noreferrer"
                        style={{ color: 'var(--text-primary)', fontWeight: 600, textDecoration: 'none' }}
                        title={item.title}
                      >
                        {item.title}
                      </a>
                      <div className="small faint mono" style={{ fontSize: 11 }}>
                        ID: {item.video_id}
                        {item.playlist_title && ` · ${item.playlist_title}`}
                      </div>
                    </div>
                  </td>
                  <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                    {formatDuration(item.duration_s)}
                  </td>
                  <td style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                    {item.actions_count > 0 ? (
                      <span className="pill small">{item.actions_count}</span>
                    ) : (
                      <span style={{ opacity: 0.4 }}>-</span>
                    )}
                  </td>
                  <td style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                    {item.grounded_count > 0 ? (
                      <span className="pill small" style={{ borderColor: 'rgba(52, 211, 153, 0.4)', color: '#34d399' }}>
                        {item.grounded_count}
                      </span>
                    ) : (
                      <span style={{ opacity: 0.4 }}>-</span>
                    )}
                  </td>
                  <td>
                    {item.error_msg ? (
                      <span className="q-error-tag" title={item.error_msg}>
                        {item.error_msg}
                      </span>
                    ) : item.status === 'completed' ? (
                      <span className="small faint" style={{ color: '#34d399' }}>
                        Joined to unified.parquet
                      </span>
                    ) : item.status === 'downloading' ? (
                      <span className="small faint" style={{ color: '#fbbf24' }}>
                        Downloading video stream...
                      </span>
                    ) : item.status === 'trajectory' ? (
                      <span className="small faint" style={{ color: '#38bdf8' }}>
                        Extracting trajectory...
                      </span>
                    ) : item.status === 'grounding' ? (
                      <span className="small faint" style={{ color: '#c084fc' }}>
                        Grounding coordinates...
                      </span>
                    ) : (
                      <span className="small faint" style={{ opacity: 0.5 }}>
                        Queued in SQLite
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
