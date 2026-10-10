import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, type Row, type RowsResponse } from '../api'

/**
 * The unified dataset as an 18-column table, plus the checks that say whether it can be trusted.
 *
 * Single unified dataset (no train/validation splitting):
 *  - 11 parent columns matching GUI-360-word-unsloth
 *  - 7 provenance columns
 *
 * Features horizontal scrolling, customizable column visibility presets, live search,
 * and a deep-inspection modal for structured fields (prompt, messages, bbox, answer).
 */

interface ColumnMeta {
  key: string
  label: string
  category: 'parent' | 'provenance'
  description: string
}

const ALL_COLUMNS: ColumnMeta[] = [
  { key: 'id', label: 'id', category: 'parent', description: 'Unique identifier per grounded action' },
  { key: 'trajectory', label: 'trajectory', category: 'parent', description: 'Trajectory identifier' },
  { key: 'sub_task', label: 'sub_task', category: 'provenance', description: 'High-level user task goal' },
  { key: 'instruction', label: 'instruction', category: 'parent', description: 'Specific action instruction' },
  { key: 'action_type', label: 'action_type', category: 'provenance', description: 'Action primitive (click, dragTo, etc.)' },
  { key: 'timestamp_s', label: 'timestamp_s', category: 'provenance', description: 'Timestamp in video seconds' },
  { key: 'target_x', label: 'target_x', category: 'parent', description: 'Target X coordinate (pixels)' },
  { key: 'target_y', label: 'target_y', category: 'parent', description: 'Target Y coordinate (pixels)' },
  { key: 'bbox', label: 'bbox', category: 'provenance', description: 'Bounding box [x1, y1, x2, y2]' },
  { key: 'image_width', label: 'image_width', category: 'parent', description: 'Frame width (pixels)' },
  { key: 'image_height', label: 'image_height', category: 'parent', description: 'Frame height (pixels)' },
  { key: 'source_video', label: 'source_video', category: 'provenance', description: 'Original MP4 video file' },
  { key: 'task_id', label: 'task_id', category: 'provenance', description: 'Task index within trajectory' },
  { key: 'step', label: 'step', category: 'provenance', description: 'Step index within task' },
  { key: 'prompt', label: 'prompt', category: 'parent', description: 'Fine-tuning system prompt' },
  { key: 'answer_raw', label: 'answer_raw', category: 'parent', description: 'Raw coordinate string' },
  { key: 'image', label: 'image', category: 'parent', description: 'Image dict/path reference' },
  { key: 'messages', label: 'messages', category: 'parent', description: 'Conversational messages structure' },
]

export function ResultsTab({ work }: { work: string }) {
  const [data, setData] = useState<RowsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [visibleCols, setVisibleCols] = useState<Set<string>>(
    () => new Set(ALL_COLUMNS.map((c) => c.key))
  )
  const [showColPicker, setShowColPicker] = useState(false)
  const [inspectModal, setInspectModal] = useState<{ title: string; content: string } | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const rows = await api.rows(work)
      setData(rows)
      setError(null)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setLoading(false)
    }
  }, [work])


  useEffect(() => {
    void load()
  }, [load])

  // Filter rows by search string
  const filteredRows = useMemo(() => {
    if (!data?.rows) return []
    let rows = data.rows

    if (search.trim()) {
      const q = search.trim().toLowerCase()
      rows = rows.filter(
        (r) =>
          r.instruction?.toLowerCase().includes(q) ||
          r.id?.toLowerCase().includes(q) ||
          r.action_type?.toLowerCase().includes(q) ||
          r.source_video?.toLowerCase().includes(q) ||
          r.sub_task?.toLowerCase().includes(q) ||
          r.trajectory?.toLowerCase().includes(q)
      )
    }

    return rows
  }, [data, search])

  const toggleCol = (key: string) => {
    setVisibleCols((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        if (next.size > 1) next.delete(key) // keep at least 1 column
      } else {
        next.add(key)
      }
      return next
    })
  }

  const setPreset = (type: 'all' | 'parent' | 'provenance' | 'compact') => {
    if (type === 'all') {
      setVisibleCols(new Set(ALL_COLUMNS.map((c) => c.key)))
    } else if (type === 'parent') {
      setVisibleCols(new Set(ALL_COLUMNS.filter((c) => c.category === 'parent').map((c) => c.key)))
    } else if (type === 'provenance') {
      setVisibleCols(new Set(['id', ...ALL_COLUMNS.filter((c) => c.category === 'provenance').map((c) => c.key)]))
    } else if (type === 'compact') {
      setVisibleCols(new Set(['id', 'instruction', 'action_type', 'timestamp_s', 'target_x', 'target_y', 'image_width', 'image_height']))
    }
  }

  const copyText = (text: string, id: string) => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopiedId(id)
      setTimeout(() => setCopiedId(null), 1500)
    })
  }

  if (loading) {
    return (
      <div className="body">
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
      <div className="body">
        <div className="col">
          <div className="notice error">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            <span>
              Could not load the dataset: {error}
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

  if (!data || data.count === 0) {
    return (
      <div className="body">
        <div className="col">
          <div className="card">
            <h2>No rows yet</h2>
            <p className="muted">
              Run steps 0 through 4, then step 5 to make sure every screenshot is
              present. A row exists only if its action grounded AND its frame is on
              disk — a missing screenshot silently removes the row.
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

  const fatal = data.problems.filter((p) => !p.startsWith('NOTE'))
  const totalCount = data.total_count ?? data.count
  const activeColsList = ALL_COLUMNS.filter((c) => visibleCols.has(c.key))

  return (
    <div className="body">
      <div className="col">
        <div className="card">
          <h2>Unified Dataset Summary</h2>
          <div className="progress-head">
            <span className="now" style={{ fontSize: 16 }}>
              {totalCount} validated row{totalCount === 1 ? '' : 's'} in unified dataset
            </span>
            <button className="btn ghost small" onClick={() => void load()}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.19" />
              </svg>
              Reload
            </button>
          </div>

          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 4 }}>
            <span className="pill small">
              dataset: <b>unified</b>
            </span>
            <span className="pill small">
              total rows: <b>{totalCount}</b>
            </span>
            <span className="pill small">
              schema columns: <b>18</b>
            </span>
          </div>

          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: 11.5, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 8 }}>
              Parquet Export Artifact
            </div>
            <div className="parquet-card highlight">
              <div className="parquet-card-head">
                <span className="parquet-card-title">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <line x1="16" y1="13" x2="8" y2="13" />
                    <line x1="16" y1="17" x2="8" y2="17" />
                    <polyline points="10 9 9 9 8 9" />
                  </svg>
                  unified.parquet
                </span>
                <span className="split-badge unified">Unified Dataset Only</span>
              </div>
              <div className="parquet-card-count">{totalCount} rows</div>
              <div className="parquet-card-desc">
                Single unified dataset of all {totalCount} grounded actions across all trajectories. Separate train and validation partitions have been removed.
              </div>
            </div>
          </div>
        </div>

        {fatal.length > 0 && (
          <div className="notice error">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            <span>
              <b>{fatal.length} verification problem(s).</b> Treat rows as
              untrustworthy until these are resolved.
              <div className="mono small" style={{ marginTop: 6 }}>
                {fatal.slice(0, 6).map((p) => (
                  <div key={p}>{p}</div>
                ))}
              </div>
            </span>
          </div>
        )}
      </div>

      <div className="col">
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
            <div>
              <h2 style={{ marginBottom: 4 }}>
                <span>Unified Dataset Rows (All 18 Parquet Columns)</span>
              </h2>
              <p className="small faint" style={{ margin: 0 }}>
                Single unified dataset table. 11 parent columns from <span className="mono">GUI-360-word-unsloth</span> + 7 provenance columns.
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button
                className={`btn small ${showColPicker ? 'ghost' : ''}`}
                onClick={() => setShowColPicker((v) => !v)}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 3h7a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-7m0-18H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h7m0-18v18" />
                </svg>
                <span>Columns ({visibleCols.size}/18)</span>
              </button>
            </div>
          </div>

          {showColPicker && (
            <div className="col-chips-wrap">
              <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)' }}>
                  Toggle Visible Columns
                </span>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button className="cell-btn" onClick={() => setPreset('all')}>All 18</button>
                  <button className="cell-btn" onClick={() => setPreset('parent')}>11 Parent</button>
                  <button className="cell-btn" onClick={() => setPreset('provenance')}>7 Provenance</button>
                  <button className="cell-btn" onClick={() => setPreset('compact')}>Compact</button>
                </div>
              </div>

              {ALL_COLUMNS.map((col) => {
                const isActive = visibleCols.has(col.key)
                return (
                  <button
                    key={col.key}
                    className={`col-chip ${isActive ? `active ${col.category}` : ''}`}
                    onClick={() => toggleCol(col.key)}
                    title={col.description}
                  >
                    <span>{isActive ? '✓' : '+'}</span>
                    <span>{col.label}</span>
                    <span style={{ opacity: 0.6, fontSize: 9 }}>({col.category})</span>
                  </button>
                )
              })}
            </div>
          )}

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, margin: '14px 0 10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input
                type="text"
                className="table-filter-input"
                placeholder="Search instructions, sub_task, action, ID..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{ minWidth: 320 }}
              />
              {search && (
                <button
                  className="btn ghost small"
                  onClick={() => setSearch('')}
                  title="Clear filter"
                >
                  ✕ Clear
                </button>
              )}
            </div>

            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Showing <b>{filteredRows.length}</b> of <b>{totalCount}</b> unified rows · <b>{visibleCols.size}</b> active columns
            </div>
          </div>

          {filteredRows.length === 0 ? (
            <div style={{ padding: '36px 16px', textAlign: 'center', color: 'var(--text-muted)' }}>
              <p>No rows match the current search filter.</p>
              <button
                className="btn small"
                onClick={() => setSearch('')}
              >
                Clear search filter
              </button>
            </div>
          ) : (
            <div className="table-wrap parquet-full">
              <table className="data">
                <thead>
                  <tr>
                    {activeColsList.map((col) => (
                      <th
                        key={col.key}
                        className={`${col.category}-col`}
                        title={col.description}
                      >
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                          <span>{col.label}</span>
                          <span style={{ fontSize: 9, opacity: 0.6, textTransform: 'lowercase', fontWeight: 400 }}>
                            {col.category}
                          </span>
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.map((row, index) => {
                    const rowKey = row.id ?? `row-${index}`

                    return (
                      <tr key={rowKey}>
                        {activeColsList.map((col) => (
                          <td key={col.key}>
                            {renderCell(col.key, row, (title, content) => {
                              setInspectModal({ title, content })
                            }, (text, id) => copyText(text, id), copiedId)}
                          </td>
                        ))}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {inspectModal && (
        <div className="inspector-overlay" onClick={() => setInspectModal(null)}>
          <div className="inspector-card" onClick={(e) => e.stopPropagation()}>
            <div className="inspector-head">
              <span style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-primary)' }}>
                {inspectModal.title}
              </span>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="btn small"
                  onClick={() => copyText(inspectModal.content, 'modal')}
                >
                  {copiedId === 'modal' ? '✓ Copied' : 'Copy'}
                </button>
                <button className="btn ghost small" onClick={() => setInspectModal(null)}>
                  ✕ Close
                </button>
              </div>
            </div>
            <pre className="inspector-body">{inspectModal.content}</pre>
          </div>
        </div>
      )}
    </div>
  )
}

function renderCell(
  key: string,
  row: Row,
  onInspect: (title: string, content: string) => void,
  onCopy: (text: string, id: string) => void,
  copiedId: string | null
) {
  switch (key) {
    case 'id':
      return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="mono" style={{ color: 'var(--brand-teal)', fontWeight: 600 }}>
            {row.id}
          </span>
          <button
            className="cell-btn"
            style={{ padding: '1px 5px', fontSize: 10 }}
            onClick={() => onCopy(row.id ?? '', row.id ?? '')}
            title="Copy ID"
          >
            {copiedId === row.id ? '✓' : 'copy'}
          </button>
        </div>
      )

    case 'trajectory':
      return (
        <div className="cell-truncate mono small" title={row.trajectory}>
          {row.trajectory}
        </div>
      )

    case 'sub_task':
      return (
        <div
          className="cell-truncate"
          style={{ maxWidth: 220, cursor: 'pointer' }}
          title={row.sub_task}
          onClick={() => onInspect(`sub_task: ${row.id}`, row.sub_task ?? '')}
        >
          {row.sub_task || '—'}
        </div>
      )

    case 'instruction':
      return (
        <div
          className="cell-truncate"
          style={{ maxWidth: 280, fontWeight: 500, color: 'var(--text-primary)' }}
          title={row.instruction}
          onClick={() => onInspect(`instruction: ${row.id}`, row.instruction ?? '')}
        >
          {row.instruction}
        </div>
      )

    case 'action_type':
      return (
        <span className="pill small" style={{ padding: '2px 8px', fontSize: 11 }}>
          {row.action_type}
        </span>
      )

    case 'timestamp_s':
      return (
        <span className="mono num">
          {row.timestamp_s !== undefined ? `${row.timestamp_s.toFixed(1)}s` : '—'}
        </span>
      )

    case 'target_x':
      return <span className="mono num">{row.target_x}</span>

    case 'target_y':
      return <span className="mono num">{row.target_y}</span>

    case 'bbox': {
      const bStr = typeof row.bbox === 'string' ? row.bbox : JSON.stringify(row.bbox)
      return (
        <span
          className="mono small faint"
          style={{ cursor: 'pointer' }}
          title="Click to inspect bbox"
          onClick={() => onInspect(`bbox: ${row.id}`, bStr || '—')}
        >
          {bStr || '—'}
        </span>
      )
    }

    case 'image_width':
      return <span className="mono num faint">{row.image_width}</span>

    case 'image_height':
      return <span className="mono num faint">{row.image_height}</span>

    case 'source_video':
      return (
        <div className="cell-truncate mono small faint" style={{ maxWidth: 200 }} title={row.source_video}>
          {row.source_video}
        </div>
      )

    case 'task_id':
      return <span className="mono num faint">#{row.task_id}</span>

    case 'step':
      return <span className="mono num faint">#{row.step}</span>

    case 'prompt': {
      const p = row.prompt ?? ''
      return (
        <button
          className="cell-btn"
          onClick={() => onInspect(`prompt: ${row.id}`, p)}
          title="Click to view full prompt text"
        >
          Prompt ({p.length} chars)
        </button>
      )
    }

    case 'answer_raw': {
      const ans = row.answer_raw ?? ''
      return (
        <code
          style={{
            color: 'var(--brand-teal)',
            background: 'rgba(0, 229, 185, 0.08)',
            padding: '2px 6px',
            borderRadius: 4,
            cursor: 'pointer',
            fontSize: 11,
          }}
          onClick={() => onInspect(`answer_raw: ${row.id}`, ans)}
          title="Click to view full answer"
        >
          {ans.replace(/\n/g, ' ')}
        </code>
      )
    }

    case 'image': {
      const p = row.image_path || (typeof row.image === 'object' && row.image?.path ? row.image.path : '')
      return (
        <button
          className="cell-btn"
          onClick={() => onInspect(`image path: ${row.id}`, p)}
          title={p}
        >
          Frame ({row.image_width}×{row.image_height})
        </button>
      )
    }

    case 'messages': {
      const msgs = row.messages ?? []
      return (
        <button
          className="cell-btn"
          onClick={() => onInspect(`messages JSON: ${row.id}`, JSON.stringify(msgs, null, 2))}
          title="Click to view conversation JSON"
        >
          {Array.isArray(msgs) ? `${msgs.length} turns` : 'Messages JSON'}
        </button>
      )
    }

    default: {
      const val = (row as Record<string, unknown>)[key]
      return <span className="small faint">{String(val ?? '—')}</span>
    }
  }
}

