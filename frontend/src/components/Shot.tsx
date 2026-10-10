import { useState } from 'react'
import { api, type Row } from '../api'

/**
 * One screenshot with the target marked.
 *
 * Two views, and the difference matters:
 *
 *  - `full` shows the whole frame. A crop proves the ring landed on *a* button;
 *    only the whole frame shows whether it was the RIGHT one, because the
 *    surrounding menus and panels are what make an instruction readable.
 *  - `zoom` shows a 320px window at 2x, which is better for judging precision to
 *    the pixel than a 1920x1080 image squeezed into a column.
 *
 * Shared by the Results and Details tabs rather than duplicated, so a fix to the
 * marker cannot land in one and miss the other.
 */

const CROP_W = 320
const CROP_H = 176
const CROP_SCALE = 2

export type ShotView = 'full' | 'zoom'

function extractMessagesText(row: Row): string {
  const msgs: unknown = row.messages
  if (Array.isArray(msgs) && msgs.length > 0) {
    const parts: string[] = []
    for (const m of msgs) {
      const msgObj = m as { role?: string; content?: unknown }
      const roleLabel = msgObj.role ? `[${msgObj.role}] ` : ''
      if (typeof msgObj.content === 'string' && msgObj.content.trim()) {
        parts.push(`${roleLabel}${msgObj.content.trim()}`)
      } else if (Array.isArray(msgObj.content)) {
        const textParts: string[] = []
        for (const c of msgObj.content) {
          if (c && typeof c === 'object' && 'text' in c) {
            const txt = (c as { text: unknown }).text
            if (typeof txt === 'string' && txt.trim()) {
              textParts.push(txt.trim())
            }
          }
        }
        if (textParts.length > 0) {
          parts.push(`${roleLabel}${textParts.join('\n')}`)
        }
      } else if (msgObj.content && typeof msgObj.content === 'object') {
        parts.push(`${roleLabel}${JSON.stringify(msgObj.content)}`)
      }
    }
    if (parts.length > 0) {
      return parts.join('\n\n')
    }
  } else if (typeof msgs === 'string' && msgs.trim()) {
    try {
      const parsed: unknown = JSON.parse(msgs)
      if (Array.isArray(parsed)) {
        return extractMessagesText({ ...row, messages: parsed })
      }
    } catch {
      return msgs.trim()
    }
  }

  // Fallback to prompt/answer if messages column is empty
  if (row.prompt && row.prompt.trim()) {
    return row.answer_raw
      ? `[user] ${row.prompt.trim()}\n\n[assistant] ${row.answer_raw.trim()}`
      : `[user] ${row.prompt.trim()}`
  }

  return ''
}

function ShotCaptionContent({ row }: { row: Row }) {
  const [expanded, setExpanded] = useState(false)
  const x = row.target_x ?? 0
  const y = row.target_y ?? 0
  const messagesText = extractMessagesText(row)

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6, flexWrap: 'wrap', gap: 6 }}>
        <span style={{ fontWeight: 700, color: 'var(--brand-teal)' }}>{row.id}</span>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {row.timestamp_s !== undefined && (
            <span className="pill small" style={{ padding: '1px 7px', fontSize: 11 }}>
              t={row.timestamp_s.toFixed(1)}s
            </span>
          )}
          {row.action_type && (
            <span className="pill small" style={{ padding: '1px 7px', fontSize: 11, color: 'var(--brand-blue)' }}>
              {row.action_type}
            </span>
          )}
          <span className="pill small" style={{ padding: '1px 7px', fontSize: 11, color: '#fca5a5' }}>
            ({x}, {y})
          </span>
        </div>
      </div>

      <div style={{ color: 'var(--text-primary)', fontSize: 12.5, lineHeight: 1.45 }}>
        {row.instruction ?? ''}
      </div>

      {messagesText && (
        <div
          style={{
            marginTop: 8,
            paddingTop: 8,
            borderTop: '1px solid rgba(255, 255, 255, 0.08)',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: 4,
            }}
          >
            <span
              style={{
                fontSize: 10.5,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: 'var(--text-muted)',
                fontWeight: 650,
              }}
            >
              Message
            </span>
            <button
              type="button"
              onClick={() => setExpanded(!expanded)}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--brand-teal)',
                fontSize: 11,
                cursor: 'pointer',
                padding: '2px 6px',
                borderRadius: 4,
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                fontWeight: 600,
                transition: 'background 0.15s ease',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(45, 212, 191, 0.1)')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'none')}
            >
              <span>{expanded ? 'Show less' : 'Expand'}</span>
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{
                  transform: expanded ? 'rotate(180deg)' : 'none',
                  transition: 'transform 0.2s ease',
                }}
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
          </div>
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              lineHeight: 1.45,
              color: 'var(--text-secondary)',
              background: 'rgba(0, 0, 0, 0.35)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              padding: '6px 8px',
              borderRadius: 4,
              whiteSpace: expanded ? 'pre-wrap' : 'normal',
              wordBreak: 'break-word',
              maxHeight: expanded ? '300px' : '40px',
              overflowY: expanded ? 'auto' : 'hidden',
              display: expanded ? 'block' : '-webkit-box',
              WebkitLineClamp: expanded ? undefined : 2,
              WebkitBoxOrient: expanded ? undefined : 'vertical',
              textOverflow: 'ellipsis',
            }}
            title={expanded ? undefined : 'Click Expand to view full dataset message'}
          >
            {messagesText}
          </div>
        </div>
      )}
    </>
  )
}

export function Shot({ row, view }: { row: Row; view: ShotView }) {
  const w = row.image_width ?? 0
  const h = row.image_height ?? 0
  const x = row.target_x ?? 0
  const y = row.target_y ?? 0
  const hasFrame = Boolean(row.image_path) && w > 0 && h > 0

  if (!hasFrame) {
    return (
      <div className="crop">
        <div className="cap">
          {row.id}: no frame on disk. Run step 5 to recover it from the original
          video.
        </div>
      </div>
    )
  }

  if (view === 'full') {
    return (
      <figure className="shot">
        <div className="shot-frame">
          <img
            src={api.frameUrl(row.image_path!)}
            alt={row.instruction ?? row.id ?? 'frame'}
            loading="lazy"
          />
          {/* Positioned in percentages, so the marker stays on the same pixel at
              any display width -- including after a window resize. */}
          <span
            className="ring"
            style={{ left: `${(x / w) * 100}%`, top: `${(y / h) * 100}%` }}
          />
        </div>
        <figcaption className="cap">
          <ShotCaptionContent row={row} />
        </figcaption>
      </figure>
    )
  }

  return (
    <div className="crop">
      <div
        style={{
          width: '100%',
          aspectRatio: `${CROP_W} / ${CROP_H}`,
          backgroundImage: `url("${api.frameUrl(row.image_path!)}")`,
          backgroundSize: `${w * CROP_SCALE}px ${h * CROP_SCALE}px`,
          backgroundPosition: `${-(x * CROP_SCALE - CROP_W / 2)}px ${-(y * CROP_SCALE - CROP_H / 2)}px`,
          backgroundRepeat: 'no-repeat',
          position: 'relative',
        }}
      >
        <span className="crop-ring" />
      </div>
      <div className="cap">
        <ShotCaptionContent row={row} />
      </div>
    </div>
  )
}
