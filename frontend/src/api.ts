/**
 * Types and fetch helpers for the API.
 *
 * The shapes mirror `jobs.Runner.snapshot` and the artefact endpoints exactly. If
 * one drifts the compiler will not catch it, so the backend is deliberately the
 * only place these are produced -- nothing here invents a field.
 */

export type StepKey =
  | 'screen'
  | 'download'
  | 'trajectory'
  | 'grounding'
  | 'dataset'
  | 'review'
  | 'autoflow'
  | 'playlist'

export interface StepInfo {
  key: StepKey
  label: string
  help: string
  done: boolean
  is_next: boolean
  is_running: boolean
  /** A step opens only once its predecessor has run. */
  locked: boolean
  /** Empty when unlocked; otherwise the step to run first. */
  locked_reason: string
}

export type Level = 'phase' | 'task' | 'ok' | 'info' | 'warn' | 'error'

export interface ActivityItem {
  level: Level
  message: string
  at: number
}

export interface Progress {
  /** 0-100, or null when the step is working but has no number to report yet. */
  pct: number | null
  detail: string
}

export interface RunState {
  work: string
  running: boolean
  step: StepKey | null
  label: string
  error: string | null
  result: Record<string, unknown> | null
  progress: Progress | null
  action: string
  activity: ActivityItem[]
  activity_total: number
  log: string
  started: number | null
  elapsed: number
  steps: StepInfo[]
  next: StepKey | 'done'
  next_unlocked: boolean
}

export interface Row {
  id?: string
  split?: string
  trajectory?: string
  sub_task?: string
  instruction?: string
  action_type?: string
  timestamp_s?: number
  target_x?: number
  target_y?: number
  bbox?: string | number[]
  image_width?: number
  image_height?: number
  image_path?: string
  image?: { bytes?: string | null; path?: string } | string
  prompt?: string
  answer_raw?: string
  messages?: Array<{ role: string; content: unknown }>
  source_video?: string
  task_id?: number
  step?: number
  point_norm?: number[]
}

export interface Tokens {
  input: number | null
  output: number | null
  thinking: number | null
  total: number | null
}

export interface ClipDetail {
  index: number
  start_s: number
  end_s: number
  seconds: number
  model: string
  tasks: number | null
  actions: number | null
  shots: number
  reply_chars: number | null
  cached: boolean
  tokens: Tokens
  cost_usd: number | null
  elapsed_s: number | null
  error?: string | null
}

export interface VideoDetail {
  stem: string
  video: string
  duration_s: number | null
  width: number | null
  height: number | null
  fps: number | null
  clip_seconds: number | null
  trajectory_model: string | null
  grounding_model: string | null
  prompt_version: string | null
  stats: Record<string, number>
  slicing: {
    clip_count: number
    clips: ClipDetail[]
    total_clip_seconds: number
    /** null for records written before truncation was tracked -- not false. */
    truncated: boolean | null
    planned_clips: number | null
    processed_clips: number | null
    covered_s: number | null
    covers_pct: number | null
  }
  actions: {
    total: number
    by_type: Record<string, number>
    tasks: number
    groundable: number
    grounded: number
    dropped: number
    yield_pct: number | null
    avg_attempts: number | null
    recovered_frames: number
    /** Groundable actions before any cap. null for records that predate the field. */
    available: number | null
    /** The cap actually applied, or null when nothing was capped. */
    limited_to: number | null
  }
  tokens: {
    trajectory: Tokens
    grounding: Tokens
    total: Tokens
    trajectory_usd: number | null
    grounding_usd: number | null
    total_usd: number | null
    unknown: string[]
  }
  elapsed: {
    trajectory_s: number | null
    grounding_s: number | null
    recorded: boolean
  }
  screening: {
    title: string | null
    accepted: boolean | null
    subtitle_chars: number | null
    subtitle_language: string | null
    subtitle_kind: string | null
    scores: Record<string, number> | null
  } | null
}

export interface DetailsResponse {
  work: string
  videos: VideoDetail[]
  video_count: number
  totals: {
    tokens: Tokens
    usd: number | null
    clips: number
    actions: number
    grounded: number
    unknown_sources: string[]
    complete: boolean
    truncated: Array<{
      stem: string
      processed: number | null
      planned: number | null
      covers_pct: number | null
    }>
    /** Videos where grounding was capped, so some groundable actions are absent. */
    limited: Array<{
      stem: string
      attempted: number | null
      available: number | null
    }>
  }
}

export interface RowsResponse {
  count: number
  total_count?: number
  columns?: string[]
  rows: Row[]
  splits: Record<string, number>
  problems: string[]
}

export interface QueueItem {
  video_id: string
  playlist_id: string | null
  playlist_title: string | null
  index_in_list: number
  title: string
  url: string
  duration_s: number | null
  status: 'pending' | 'downloading' | 'trajectory' | 'grounding' | 'completed' | 'failed'
  error_msg: string | null
  local_video_path: string | null
  actions_count: number
  grounded_count: number
  created_at: string
  updated_at: string
}

export interface QueueResponse {
  items: QueueItem[]
  total: number
  stats: Record<string, number>
}

export interface UrlInspectionItem {
  video_id: string
  title: string
  url: string
  duration_s: number | null
  index_in_list: number
}

export interface UrlInspectionData {
  is_playlist: boolean
  playlist_id: string | null
  playlist_title: string | null
  channel?: string | null
  total_videos: number
  items: UrlInspectionItem[]
}

export interface UrlInspectionResponse {
  ok: boolean
  data: UrlInspectionData
}

export interface PromptEntry {
  name: string
  title: string
  kind: string
  source: string | null
  model_const: string | null
  used_by: string | null
  notes: string | null
  text: string
  chars: number
}

export interface ScreenRecord {
  title?: string
  duration_s?: number
  resolution?: [number, number]
  subtitle_chars?: number
  subtitle_source?: string
  accepted?: boolean
  /** The paper's candidate rule, evaluated at screening. */
  too_long?: boolean
  duration_limit_s?: number
  /** Why the verdict came out as it did, one line per criterion. */
  reasons?: string[]
  meta_verdict?: {
    is_gui_content?: boolean
    confidence?: number
    reasoning?: string
  }
  quality?: {
    scores?: Record<string, number>
    reasons?: Record<string, string>
    blind_dimensions?: string[]
  }
}

export interface ReviewVideo {
  checked: number
  recovered: number
  missing: number
  no_source: boolean
}

export interface ReviewReport {
  frames_checked: number
  frames_present: number
  frames_missing: number
  frames_recovered: number
  frames_unrecoverable: number
  unrecoverable: string[]
  coverage: number
  rows: number
  dataset_stale: boolean
  videos: Record<string, ReviewVideo>
}

export interface ThinkingLevel {
  value: string
  note: string
}

export interface Options {
  thinking_levels: ThinkingLevel[]
  default_thinking: string
  default_model: Record<string, string>
}

export interface Health {
  ok: boolean
  web_built: boolean
  key_source: string | null
  key_tail: string | null
  /** Step keys this server knows about. Missing ones mean stale code. */
  steps: string[]
  started: number
  uptime_s: number
  newest_source: number
  /** True when a module on disk was edited after this process started. */
  stale_code: boolean
  /** The bundle the server would serve now. Compare with the one loaded. */
  web_asset: string
}

export interface SchemaColumn {
  name: string
  type: string
  /** True when the column also exists in the parent dataset. */
  from_parent: boolean
  /** How many rows hold a value here, or null when not computed. */
  populated: number | null
}

export interface DatasetSchema {
  columns: SchemaColumn[]
  count: number
  parent_columns: string[]
  parent_count: number
  extra: string[]
  order_matches_parent: boolean
  rows: number | null
  splits: Record<string, number>
  total_rows: number
  problems: string[]
}

async function get<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`)
  }
  return (await response.json()) as T
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`)
  }
  return (await response.json()) as T
}

export const api = {
  state: (work: string) => get<RunState>(`/api/state?work=${encodeURIComponent(work)}`),

  run: (key: StepKey, payload: Record<string, unknown>) =>
    post<{ started: boolean; reason?: string; message: string }>(
      `/api/run/${key}`,
      payload,
    ),

  reset: (work?: string) =>
    post<{ ok: boolean }>(`/api/reset?work=${encodeURIComponent(work || 'wilcom_work')}`, {}),

  rows: (work: string) =>
    get<RowsResponse>(`/api/rows?work=${encodeURIComponent(work)}`),


  review: (work: string) =>
    get<{ report: ReviewReport | null }>(`/api/review?work=${encodeURIComponent(work)}`),

  details: (work: string) =>
    get<DetailsResponse>(`/api/details?work=${encodeURIComponent(work)}`),

  options: () => get<Options>('/api/options'),

  schema: (work: string) =>
    get<DatasetSchema>(`/api/schema?work=${encodeURIComponent(work)}`),

  health: () => get<Health>('/api/health'),

  screening: (work: string) =>
    get<{ record: ScreenRecord | null }>(`/api/screen?work=${encodeURIComponent(work)}`),

  prompts: () => get<{ prompts: PromptEntry[]; count: number }>('/api/prompts'),

  plan: (seconds: number) => get<Record<string, unknown>>(`/api/plan?seconds=${seconds}`),

  frameUrl: (path: string) => `/api/frame?path=${encodeURIComponent(path)}`,

  queue: (work: string) => get<QueueResponse>(`/api/queue?work=${encodeURIComponent(work)}`),

  enqueue: (url: string, work: string) =>
    post<{ ok: boolean; result: unknown }>('/api/queue/enqueue', { url, work }),

  runQueue: (work: string, model_traj?: string, model_ground?: string, thinking?: string) =>
    post<{ started: boolean; step: string; message: string }>('/api/queue/run', {
      work,
      model_traj: model_traj || '',
      model_ground: model_ground || '',
      thinking: thinking || '',
    }),

  retryQueue: (work: string) =>
    post<{ ok: boolean; reset_count: number }>('/api/queue/retry', { work }),

  clearQueue: (work: string, mode: 'all' | 'finished' = 'all') =>
    post<{ ok: boolean; cleared_count: number }>('/api/queue/clear', { work, mode }),

  inspectUrl: (url: string) =>
    post<UrlInspectionResponse>('/api/url/inspect', { url }),

  runAutoflow: (payload: {
    work?: string
    url?: string
    source?: string
    thinking?: string
    max_clips?: number
    model_traj?: string
    model_ground?: string
  }) =>
    post<{ started: boolean; reason?: string; message: string }>('/api/run/autoflow', payload),

  autoflowQueue: (payload: {
    url?: string
    work?: string
    model_traj?: string
    model_ground?: string
    thinking?: string
  }) =>
    post<{ started: boolean; reason?: string; step?: string; message: string; enqueued?: unknown }>(
      '/api/queue/autoflow',
      payload,
    ),
}

