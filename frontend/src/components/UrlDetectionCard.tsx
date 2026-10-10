import { useEffect, useRef, useState } from 'react'
import { api, type UrlInspectionData } from '../api'

interface UrlDetectionCardProps {
  url: string
  currentMode: 'step' | 'playlist'
  stepName?: string
  inspectTrigger?: number
  onSwitchToPlaylist?: (url: string) => void
  onSwitchToStep?: (videoUrl: string) => void
  onSelectSingleVideo?: (videoUrl: string) => void
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || seconds <= 0) return '--:--'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function detectLinkType(rawUrl: string): {
  isLikely: boolean
  isLikelyPlaylist: boolean
  isLikelyVideo: boolean
  hasBoth: boolean
  videoId?: string
  playlistId?: string
} {
  const trimmed = rawUrl.trim()
  if (!trimmed || (!trimmed.startsWith('http://') && !trimmed.startsWith('https://') && !trimmed.includes('youtu'))) {
    return { isLikely: false, isLikelyPlaylist: false, isLikelyVideo: false, hasBoth: false }
  }

  // Check for playlist indicators
  const hasListParam = /[?&]list=([a-zA-Z0-9_-]+)/i.exec(trimmed)
  const isPlaylistPath = /\/playlist/i.test(trimmed)
  const isLikelyPlaylist = Boolean(hasListParam || isPlaylistPath)

  // Check for video indicators
  const hasVideoParam = /[?&]v=([a-zA-Z0-9_-]+)/i.exec(trimmed)
  const isShorts = /\/shorts\/([a-zA-Z0-9_-]+)/i.exec(trimmed)
  const isYouTuDotBe = /youtu\.be\/([a-zA-Z0-9_-]+)/i.exec(trimmed)
  const isLikelyVideo = Boolean(hasVideoParam || isShorts || isYouTuDotBe)

  const videoId = hasVideoParam?.[1] || isShorts?.[1] || isYouTuDotBe?.[1]
  const playlistId = hasListParam?.[1]

  return {
    isLikely: isLikelyPlaylist || isLikelyVideo,
    isLikelyPlaylist,
    isLikelyVideo,
    hasBoth: isLikelyPlaylist && isLikelyVideo,
    videoId,
    playlistId,
  }
}

export function UrlDetectionCard({
  url,
  currentMode,
  stepName,
  inspectTrigger,
  onSwitchToPlaylist,
  onSwitchToStep,
  onSelectSingleVideo,
}: UrlDetectionCardProps) {
  const [inspecting, setInspecting] = useState(false)
  const [metadata, setMetadata] = useState<UrlInspectionData | null>(null)
  const [inspectError, setInspectError] = useState<string | null>(null)
  const [showPreview, setShowPreview] = useState(false)
  const cacheRef = useRef<Map<string, UrlInspectionData>>(new Map())

  const trimmed = url.trim()
  const detection = detectLinkType(trimmed)

  useEffect(() => {
    if (!detection.isLikely) {
      setMetadata(null)
      setInspectError(null)
      setInspecting(false)
      return
    }

    if (cacheRef.current.has(trimmed)) {
      setMetadata(cacheRef.current.get(trimmed)!)
      setInspectError(null)
      setInspecting(false)
      return
    }

    let active = true
    setInspecting(true)
    setInspectError(null)

    const delayMs = inspectTrigger ? 0 : 450
    const timer = setTimeout(async () => {
      try {
        const res = await api.inspectUrl(trimmed)
        if (active) {
          if (res?.data) {
            cacheRef.current.set(trimmed, res.data)
            setMetadata(res.data)
            setInspectError(null)
          } else {
            setMetadata(null)
          }
        }
      } catch (err: unknown) {
        if (active) {
          const msg = err instanceof Error ? err.message : String(err)
          setInspectError(msg)
          setMetadata(null)
        }
      } finally {
        if (active) {
          setInspecting(false)
        }
      }
    }, delayMs)

    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [trimmed, detection.isLikely, inspectTrigger])


  if (!detection.isLikely) {
    return null
  }

  const isPlaylist = metadata ? metadata.is_playlist : detection.isLikelyPlaylist
  const singleVideo = metadata?.items?.[0]

  return (
    <div className={`url-detect-card ${isPlaylist ? 'is-playlist' : 'is-video'}`}>
      {/* Header bar */}
      <div className="url-detect-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          {isPlaylist ? (
            <span className="url-detect-tag playlist">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="8" y1="6" x2="21" y2="6" />
                <line x1="8" y1="12" x2="21" y2="12" />
                <line x1="8" y1="18" x2="21" y2="18" />
                <line x1="3" y1="6" x2="3.01" y2="6" />
                <line x1="3" y1="12" x2="3.01" y2="12" />
                <line x1="3" y1="18" x2="3.01" y2="18" />
              </svg>
              <span>Playlist Link Detected</span>
            </span>
          ) : (
            <span className="url-detect-tag video">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
                <line x1="7" y1="2" x2="7" y2="22" />
                <line x1="17" y1="2" x2="17" y2="22" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <line x1="2" y1="7" x2="7" y2="7" />
                <line x1="2" y1="17" x2="7" y2="17" />
                <line x1="17" y1="17" x2="22" y2="17" />
                <line x1="17" y1="7" x2="22" y2="7" />
              </svg>
              <span>Single Video Link Detected</span>
            </span>
          )}

          {metadata?.total_videos ? (
            <span className="url-detect-count-pill">
              {metadata.total_videos} {metadata.total_videos === 1 ? 'video' : 'videos'}
            </span>
          ) : null}

          {detection.hasBoth && (
            <span className="url-detect-hint-pill" title="URL specifies both a video and a playlist parameter">
              Video inside Playlist
            </span>
          )}

          <span className="url-detect-hint-pill" style={{ background: 'rgba(56, 189, 248, 0.12)', color: '#38bdf8', borderColor: 'rgba(56, 189, 248, 0.3)' }}>
            Auto-Flow Ready
          </span>
        </div>

        {inspecting ? (
          <div className="url-detect-status loading">
            <span className="spinner" style={{ width: 12, height: 12 }} />
            <span>Retrieving metadata...</span>
          </div>
        ) : metadata ? (
          <div className="url-detect-status verified">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            <span>Metadata Verified • Auto-Flow Ready</span>
          </div>
        ) : null}
      </div>


      {/* Body content */}
      <div className="url-detect-body">
        {inspecting && !metadata ? (
          <div className="url-detect-placeholder">
            <div className="faint small">
              Analyzing link with <span className="mono">yt-dlp</span> to retrieve metadata, title, and video list...
            </div>
          </div>
        ) : inspectError && !metadata ? (
          <div className="url-detect-error">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>Metadata inspection warning: {inspectError} (You can still proceed with execution).</span>
          </div>
        ) : metadata ? (
          <div>
            {/* Title & Channel info */}
            <div className="url-detect-title-row">
              <h4 className="url-detect-title">
                {isPlaylist ? (metadata.playlist_title || 'YouTube Playlist') : (singleVideo?.title || 'YouTube Video')}
              </h4>
            </div>

            <div className="url-detect-meta-chips">
              {metadata.channel && (
                <div className="url-detect-chip">
                  <span className="chip-label">Channel:</span>
                  <span className="chip-value">{metadata.channel}</span>
                </div>
              )}
              {isPlaylist && (
                <div className="url-detect-chip">
                  <span className="chip-label">Total Videos:</span>
                  <span className="chip-value bold">{metadata.total_videos}</span>
                </div>
              )}
              {!isPlaylist && singleVideo?.duration_s != null && (
                <div className="url-detect-chip">
                  <span className="chip-label">Duration:</span>
                  <span className="chip-value">{formatDuration(singleVideo.duration_s)}</span>
                </div>
              )}
              {metadata.playlist_id && (
                <div className="url-detect-chip mono">
                  <span className="chip-label">Playlist ID:</span>
                  <span className="chip-value">{metadata.playlist_id}</span>
                </div>
              )}
              {!isPlaylist && singleVideo?.video_id && (
                <div className="url-detect-chip mono">
                  <span className="chip-label">Video ID:</span>
                  <span className="chip-value">{singleVideo.video_id}</span>
                </div>
              )}
            </div>

            {/* If playlist, show videos preview */}
            {isPlaylist && metadata.items && metadata.items.length > 0 && (
              <div className="url-detect-preview-wrap">
                <button
                  type="button"
                  className="url-detect-preview-toggle"
                  onClick={() => setShowPreview((p) => !p)}
                >
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    style={{
                      transform: showPreview ? 'rotate(90deg)' : 'rotate(0deg)',
                      transition: 'transform 0.15s ease',
                    }}
                  >
                    <polyline points="9 18 15 12 9 6" />
                  </svg>
                  <span>
                    {showPreview ? 'Hide playlist items' : `Preview videos in playlist (${metadata.items.length})`}
                  </span>
                </button>

                {showPreview && (
                  <div className="url-detect-items-list">
                    {metadata.items.slice(0, 6).map((item, idx) => (
                      <div key={item.video_id || idx} className="url-detect-item-row">
                        <span className="url-detect-item-idx">#{item.index_in_list || idx + 1}</span>
                        <span className="url-detect-item-title" title={item.title}>
                          {item.title}
                        </span>
                        <span className="url-detect-item-dur">{formatDuration(item.duration_s)}</span>
                      </div>
                    ))}
                    {metadata.items.length > 6 && (
                      <div className="url-detect-item-more">
                        + {metadata.items.length - 6} more videos in playlist
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        ) : null}

        {/* Guidance and Contextual Call to Action */}
        <div className="url-detect-actions-bar">
          {/* Scenario 1: User pasted a playlist URL while in Step Mode */}
          {isPlaylist && currentMode === 'step' && (
            <div className="url-detect-mode-prompt">
              <div className="url-detect-prompt-text">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" style={{ flex: 'none' }}>
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="16" x2="12" y2="12" />
                  <line x1="12" y1="8" x2="12.01" y2="8" />
                </svg>
                <span>
                  You pasted a <b>multi-video playlist link</b> ({metadata?.total_videos ?? ''} videos) while in Single Video mode. Switch to{' '}
                  <b>Playlist Queue mode</b> to run all videos sequentially on their own.
                </span>
              </div>
              <div className="url-detect-prompt-buttons">
                {onSwitchToPlaylist && (
                  <button
                    type="button"
                    className="btn primary small"
                    onClick={() => onSwitchToPlaylist(trimmed)}
                  >
                    <span>Switch to Playlist Queue Mode</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <line x1="5" y1="12" x2="19" y2="12" />
                      <polyline points="12 5 19 12 12 19" />
                    </svg>
                  </button>
                )}
                {detection.hasBoth && detection.videoId && onSelectSingleVideo && (
                  <button
                    type="button"
                    className="btn ghost small"
                    onClick={() =>
                      onSelectSingleVideo(`https://www.youtube.com/watch?v=${detection.videoId}`)
                    }
                  >
                    Use selected single video only
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Scenario 2: User is in Playlist Queue Mode and pasted a playlist */}
          {isPlaylist && currentMode === 'playlist' && (
            <div className="url-detect-mode-prompt ready">
              <div className="url-detect-prompt-text">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ flex: 'none' }}>
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                  <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
                <span>
                  <b>Metadata verified:</b> Found <b>{metadata?.playlist_title || 'YouTube Playlist'}</b> with{' '}
                  <b>{metadata?.total_videos || 'multiple'} videos</b>. Ready to auto-flow sequentially into SQLite queue.
                </span>
              </div>
            </div>
          )}


          {/* Scenario 3: User pasted a single video URL in Playlist Queue mode */}
          {!isPlaylist && currentMode === 'playlist' && (
            <div className="url-detect-mode-prompt info">
              <div className="url-detect-prompt-text">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" style={{ flex: 'none' }}>
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="16" x2="12" y2="12" />
                  <line x1="12" y1="8" x2="12.01" y2="8" />
                </svg>
                <span>
                  <b>Single Video detected:</b> You can enqueue it as a 1-video batch, or switch to Single Video mode.
                </span>
              </div>
              {onSwitchToStep && (
                <div className="url-detect-prompt-buttons">
                  <button
                    type="button"
                    className="btn ghost small"
                    onClick={() => onSwitchToStep(trimmed)}
                  >
                    <span>Switch to Single Video Mode</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="5" y1="12" x2="19" y2="12" />
                      <polyline points="12 5 19 12 12 19" />
                    </svg>
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Scenario 4: User is in Step Mode and pasted single video */}
          {!isPlaylist && currentMode === 'step' && (
            <div className="url-detect-mode-prompt ready">
              <div className="url-detect-prompt-text">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ flex: 'none' }}>
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                <span>
                  Single video ready for {stepName || 'pipeline execution'}: <b>{singleVideo?.title || 'YouTube Video'}</b>{' '}
                  {singleVideo?.duration_s ? `(${formatDuration(singleVideo.duration_s)})` : ''}.
                  Run the full automated pipeline or proceed step-by-step below.
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
