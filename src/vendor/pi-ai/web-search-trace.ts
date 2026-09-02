/** Opt-in, exact wire tracing for Codex hosted-search events. */

import { appendFileSync } from 'node:fs'
import { CODEX_CONNECT_VERSION } from '../../version.ts'

export const CODEX_WEB_SEARCH_TRACE_ENV = 'DSH_CODEX_WEB_SEARCH_TRACE_FILE'

type CodexTransport = 'sse' | 'websocket'
type JsonRecord = Record<string, unknown>

function record(value: unknown): JsonRecord | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as JsonRecord
    : undefined
}

function isWebSearchEvent(event: JsonRecord): boolean {
  const eventType = event['type']
  if (typeof eventType === 'string' && eventType.includes('web_search_call')) return true
  return record(event['item'])?.['type'] === 'web_search_call'
}

/**
 * Append the unmodified decoded event before normalization or validation.
 *
 * Tracing is disabled unless CODEX_WEB_SEARCH_TRACE_ENV names a file. Search
 * events can contain queries and visited URLs, so new trace files are private
 * to the current user and no unrelated response events are recorded.
 */
export function traceCodexWebSearchEvent(
  event: JsonRecord,
  transport: CodexTransport,
): void {
  const filename = process.env[CODEX_WEB_SEARCH_TRACE_ENV]?.trim()
  if (filename === undefined || filename.length === 0 || !isWebSearchEvent(event)) return

  const entry = {
    timestamp: new Date().toISOString(),
    package: 'dsh-codex-experiment',
    version: CODEX_CONNECT_VERSION,
    transport,
    event,
  }
  try {
    appendFileSync(filename, `${JSON.stringify(entry)}\n`, {
      encoding: 'utf8',
      flag: 'a',
      mode: 0o600,
    })
  } catch (cause) {
    throw new Error(`Could not write Codex web-search trace to ${JSON.stringify(filename)}`, { cause })
  }
}
