import { useEffect, useState } from 'react'
import { api, type Health } from './api'
import { useRunState } from './useRunState'
import { RunTab } from './components/RunTab'
import { ResultsTab } from './components/ResultsTab'
import { DetailsTab } from './components/DetailsTab'
import { PromptsTab } from './components/PromptsTab'

type Tab = 'run' | 'results' | 'details' | 'prompts'

const TABS: Array<{ key: Tab; label: string; icon: JSX.Element }> = [
  {
    key: 'run',
    label: 'Run',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="5 3 19 12 5 21 5 3" />
      </svg>
    ),
  },
  {
    key: 'results',
    label: 'Results',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 3h18v18H3zM3 9h18M3 15h18M9 3v18" />
      </svg>
    ),
  },
  {
    key: 'details',
    label: 'Details',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
        <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
        <line x1="12" y1="22.08" x2="12" y2="12" />
      </svg>
    ),
  },
  {
    key: 'prompts',
    label: 'Prompts',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <polyline points="10 9 9 9 8 9" />
      </svg>
    ),
  },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('run')
  const [work, setWork] = useState('wilcom_work')
  const [health, setHealth] = useState<Health | null>(null)
  const run = useRunState(work)

  // Fetched once. The two staleness signals it carries are the difference between
  // "this is broken" and "this process predates the change", which has been the
  // most confusing failure in this project.
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  // The bundle this page was loaded from, read off the DOM rather than assumed.
  //
  // Meaningless under the dev server: Vite serves `/src/main.tsx` and transpiles on
  // request, so there is no `index-XXXX.js` to compare against and the check would
  // report a stale page permanently. `import.meta.env.DEV` is how the code knows
  // which mode it is in, rather than being told.
  const loadedAsset = (() => {
    if (import.meta.env.DEV) return null
    const tag = document.querySelector('script[type="module"]')
    const src = tag?.getAttribute('src') ?? ''
    return src.split('/').pop() ?? ''
  })()

  const pageIsStale = Boolean(
    !import.meta.env.DEV &&
      health?.web_asset &&
      loadedAsset &&
      health.web_asset !== loadedAsset,
  )

  const running = run.state?.running ?? false

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <div className="brand-logo" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="4" width="20" height="16" rx="4" />
              <polygon points="10 8 16 12 10 16 10 8" fill="currentColor" stroke="none" />
              <circle cx="18" cy="7" r="2" fill="#00E5B9" stroke="none" />
            </svg>
          </div>
          <div className="brand-text">
            <span className="brand-name">WILCOM Video2GUI</span>
            <a
              href="https://arxiv.org/abs/2605.14747"
              target="_blank"
              rel="noreferrer"
              className="paper-badge"
              title="Open arXiv paper (arXiv:2605.14747)"
            >
              <span>arXiv:2605.14747</span>
              <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                <path d="M3.5 8.5L8.5 3.5M8.5 3.5H4.5M8.5 3.5V7.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </a>
          </div>
        </div>

        {running && (
          <span className="badge ok" style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
            <span className="spinner" style={{ width: 11, height: 11 }} />
            <span>{run.state?.label || 'Running pipeline step...'}</span>
          </span>
        )}

        <div className="spacer" />

        <nav className="tabs" role="tablist">
          {TABS.map((entry) => (
            <button
              key={entry.key}
              role="tab"
              aria-selected={tab === entry.key}
              onClick={() => setTab(entry.key)}
            >
              <span className="tab-icon">{entry.icon}</span>
              <span>{entry.label}</span>
            </button>
          ))}
        </nav>
      </header>

      {(health?.stale_code || pageIsStale) && (
        <div className="stale-bar">
          <span className="stale-mark">!</span>
          <span>
            {health?.stale_code && (
              <>
                <b>The server is running older code than the files on disk.</b>{' '}
                New endpoints and rules are missing, which makes them look broken.
                Stop it and start it again.
              </>
            )}
            {health?.stale_code && pageIsStale && ' '}
            {pageIsStale && (
              <>
                <b>This page is an older build.</b> Hard-reload it
                (Ctrl+Shift+R) to pick up {health?.web_asset}.
              </>
            )}
          </span>
        </div>
      )}

      {tab === 'run' && (
        <RunTab
          work={work}
          run={run}
          onChangeWork={setWork}
          onOpenResults={() => setTab('results')}
        />
      )}
      {tab === 'results' && <ResultsTab work={work} />}
      {tab === 'details' && <DetailsTab work={work} />}
      {tab === 'prompts' && <PromptsTab />}
    </div>
  )
}
