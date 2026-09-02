import { afterEach, describe, expect, it } from 'vitest'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import {
  CODEX_WEB_SEARCH_TRACE_ENV,
  traceCodexWebSearchEvent,
} from '../src/vendor/pi-ai/web-search-trace.ts'

const temporaryDirectories: string[] = []

afterEach(() => {
  delete process.env[CODEX_WEB_SEARCH_TRACE_ENV]
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true })
  }
})

function traceFilename(): string {
  const directory = mkdtempSync(join(tmpdir(), 'dsh-codex-trace-'))
  temporaryDirectories.push(directory)
  return join(directory, 'web-search.jsonl')
}

describe('Codex web-search wire trace', () => {
  it('records exact search events with the build and transport identity', () => {
    const filename = traceFilename()
    process.env[CODEX_WEB_SEARCH_TRACE_ENV] = filename
    const event = {
      type: 'response.output_item.added',
      output_index: 3,
      item: {
        type: 'web_search_call',
        id: 'ws_trace',
        status: 'in_progress',
        action: null,
      },
    }

    traceCodexWebSearchEvent(event, 'websocket')

    const line = readFileSync(filename, 'utf8').trim()
    expect(JSON.parse(line)).toEqual({
      timestamp: expect.any(String),
      package: 'dsh-codex-experiment',
      version: '0.2.8',
      transport: 'websocket',
      event,
    })
  })

  it('does not record unrelated response events', () => {
    const filename = traceFilename()
    process.env[CODEX_WEB_SEARCH_TRACE_ENV] = filename

    traceCodexWebSearchEvent({
      type: 'response.output_text.delta',
      delta: 'private assistant text',
    }, 'sse')

    expect(() => readFileSync(filename, 'utf8')).toThrow()
  })
})
