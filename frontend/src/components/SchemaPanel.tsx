import { useEffect, useState } from 'react'
import { api, type DatasetSchema } from '../api'

/**
 * The exported dataset's real columns.
 *
 * This exists because the schema was asked about twice and both times the only
 * answer was "open the parquet". Worse, the visible answer was wrong: the Results
 * tab shows a seven-column table (`id, instruction, action, t, x, y, image`), which
 * is a trimmed view for human scanning and not the schema at all -- so a correct
 * dataset looked like a broken one.
 *
 * The list is read from the exporter's own `features()`, so it cannot describe
 * something other than what is written, and the parent/ours split is a name
 * comparison rather than a claim.
 */
export function SchemaPanel({ work }: { work: string }) {
  const [data, setData] = useState<DatasetSchema | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    api
      .schema(work)
      .then((s) => {
        if (!cancelled) {
          setData(s)
          setError(null)
        }
      })
      .catch((exc) => {
        if (!cancelled) setError(exc instanceof Error ? exc.message : String(exc))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [work])

  if (loading) {
    return (
      <div className="card">
        <h2>Dataset schema</h2>
        <div className="skeleton" style={{ height: 60 }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="card">
        <h2>Dataset schema</h2>
        <div className="notice error small">
          <span>{'\u2715'}</span>
          <span>Could not read the schema: {error}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <h2>
        <span>Dataset schema</span>
        <button
          className="btn ghost small"
          onClick={() => setOpen((v) => !v)}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            {open ? (
              <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24M1 1l22 22" />
            ) : (
              <>
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
              </>
            )}
          </svg>
          <span>{open ? 'Hide columns' : `Show all ${data.count} columns`}</span>
        </button>
      </h2>

      <div className="stats">
        <Stat label="Columns" value={String(data.count)} />
        <Stat
          label="From the parent"
          value={String(data.parent_count)}
          note="GUI-360-word-unsloth"
        />
        <Stat
          label="Our additions"
          value={String(data.extra.length)}
          note="provenance only"
        />
        <Stat
          label="Rows written"
          value={data.total_rows.toLocaleString()}
          note={Object.entries(data.splits)
            .filter(([, n]) => n > 0)
            .map(([k, n]) => `${k} ${n}`)
            .join(' · ')}
        />
      </div>

      {data.problems.length > 0 ? (
        <div className="notice error" style={{ marginTop: 12 }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <span>
            <b>The parent-compatibility claim does not hold.</b>
            <div className="mono small" style={{ marginTop: 4 }}>
              {data.problems.map((p) => (
                <div key={p}>{p}</div>
              ))}
            </div>
          </span>
        </div>
      ) : (
        <div className="notice ok" style={{ marginTop: 12 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 2 }}>
            <polyline points="20 6 9 17 4 12" />
          </svg>
          <span>
            The first {data.parent_count} columns are <b>the parent's, in the
            parent's order</b>. A consumer written against{' '}
            <span className="mono">GUI-360-word-unsloth</span> reads these rows
            unchanged
            {data.extra.length > 0 &&
              `, and the ${data.extra.length} extra column${
                data.extra.length === 1 ? '' : 's'
              } at the end are ignored`}
            .
          </span>
        </div>
      )}

      {open && (
        <div className="table-wrap" style={{ maxHeight: 'none', marginTop: 14 }}>
          <table className="data">
            <thead>
              <tr>
                <th style={{ width: 40 }}>#</th>
                <th>column</th>
                <th>type</th>
                <th>origin</th>
                {data.rows !== null && <th>rows with value</th>}
              </tr>
            </thead>
            <tbody>
              {data.columns.map((column, index) => (
                <tr key={column.name}>
                  <td className="num faint">{index + 1}</td>
                  <td className="mono" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{column.name}</td>
                  <td className="mono small faint">{column.type}</td>
                  <td>
                    <span className="pill small" style={{
                      padding: '2px 8px',
                      fontSize: 11,
                      color: column.from_parent ? 'var(--ok)' : 'var(--brand-blue)',
                      borderColor: column.from_parent ? 'var(--ok-border)' : 'var(--info-border)'
                    }}>
                      {column.from_parent ? 'parent' : 'addition'}
                    </span>
                  </td>
                  {data.rows !== null && (
                    <td className="num">
                      {column.populated === null ? '—' : column.populated.toLocaleString()}
                      {column.populated === 0 && (
                        <span className="faint small"> (empty)</span>
                      )}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!open && (
        <div className="mono small muted" style={{ marginTop: 10, overflowWrap: 'anywhere' }}>
          {data.columns.map((c) => c.name).join(', ')}
        </div>
      )}

      <p className="small faint" style={{ marginTop: 10, marginBottom: 0 }}>
        The <b>Results</b> tab shows a seven-column table for human scanning. That
        is a trimmed view, not this schema — the dataset is the {data.count}{' '}
        columns above.
      </p>
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
