import { useEffect, useState } from 'react'
import { api, type PromptEntry } from '../api'

/**
 * The exact prompt text sent at each stage, with its provenance.
 *
 * This exists because the prompts are the product. An earlier version of this
 * project had three phrases silently deleted from Prompt 4 and the examples in
 * Prompt 5 narrowed, while the surrounding code claimed the prompts were verbatim.
 * Being able to read what is actually sent is the only defence against that.
 */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  const onCopy = (e: React.MouseEvent) => {
    e.stopPropagation()
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    }).catch(() => {})
  }

  return (
    <button className="btn ghost small" onClick={onCopy} title="Copy prompt text" style={{ padding: '3px 8px' }}>
      {copied ? (
        <>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--ok)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
          <span style={{ color: 'var(--ok)', fontSize: 11 }}>Copied</span>
        </>
      ) : (
        <>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
          <span style={{ fontSize: 11 }}>Copy</span>
        </>
      )}
    </button>
  )
}

export function PromptsTab() {
  const [prompts, setPrompts] = useState<PromptEntry[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState('')

  useEffect(() => {
    api
      .prompts()
      .then((result) => setPrompts(result.prompts))
      .catch((exc) => setError(exc instanceof Error ? exc.message : String(exc)))
  }, [])

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
            <span>Could not load prompts: {error}</span>
          </div>
        </div>
      </div>
    )
  }

  if (!prompts) {
    return (
      <div className="body full">
        <div className="col">
          <div className="card">
            {[0, 1, 2].map((i) => (
              <div className="skeleton" key={i} style={{ height: 46, marginBottom: 10 }} />
            ))}
          </div>
        </div>
      </div>
    )
  }

  const shown = prompts.filter((p) => {
    if (!filter.trim()) return true
    const needle = filter.toLowerCase()
    return (
      p.name.toLowerCase().includes(needle) ||
      (p.title ?? '').toLowerCase().includes(needle) ||
      p.text.toLowerCase().includes(needle)
    )
  })

  const total = shown.reduce((sum, p) => sum + p.chars, 0)

  return (
    <div className="body full">
      <div className="col">
        <div className="card">
          <h2>Paper Prompts Catalog</h2>
          <div className="field" style={{ marginBottom: 10, position: 'relative' }}>
            <input
              value={filter}
              placeholder="Search prompts by name, stage, model, or instructions..."
              onChange={(e) => setFilter(e.target.value)}
              style={{ paddingLeft: 34 }}
            />
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--text-muted)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ position: 'absolute', left: 11, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            {filter && (
              <button
                onClick={() => setFilter('')}
                style={{
                  position: 'absolute',
                  right: 10,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  fontSize: 14,
                }}
                title="Clear filter"
              >
                ✕
              </button>
            )}
          </div>
          <div className="mono small muted" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className="pill small">{shown.length} of {prompts.length} prompts</span>
            <span>·</span>
            <span className="faint">{total.toLocaleString()} total characters</span>
          </div>
        </div>

        {shown.map((prompt) => (
          <details className="prompt" key={prompt.name}>
            <summary>
              <span className="title">{prompt.title || prompt.name}</span>
              <span className="pill small" style={{ fontSize: 11 }}>{prompt.kind}</span>
              <span className="meta">
                {prompt.source ? `${prompt.source} · ` : ''}
                {prompt.chars.toLocaleString()} chars
              </span>
            </summary>
            <div style={{ padding: '14px 18px 0', borderTop: '1px solid var(--border-subtle)', background: 'rgba(255, 255, 255, 0.015)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
                <dl className="kv" style={{ margin: 0 }}>
                  <dt>key</dt>
                  <dd className="mono small" style={{ color: 'var(--brand-teal)' }}>{prompt.name}</dd>
                  <dt>used by</dt>
                  <dd className="small">{prompt.used_by || '-'}</dd>
                  <dt>model</dt>
                  <dd className="mono small">{prompt.model_const || '-'}</dd>
                  {prompt.notes && (
                    <>
                      <dt>notes</dt>
                      <dd className="small muted">{prompt.notes}</dd>
                    </>
                  )}
                </dl>
                <CopyButton text={prompt.text} />
              </div>
            </div>
            <pre>{prompt.text}</pre>
          </details>
        ))}

        {shown.length === 0 && (
          <div className="notice info">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none', marginTop: 1 }}>
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <span>No prompt matches {JSON.stringify(filter)}.</span>
          </div>
        )}
      </div>
    </div>
  )
}
