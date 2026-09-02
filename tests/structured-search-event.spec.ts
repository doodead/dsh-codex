import { describe, expect, it } from 'vitest'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import { KNOWN_SESSION_EVENT_TYPES, packChunkRuns, Session, SessionId } from '@deepseek-ai/dsh-session'
import {
  carryCodexStructuredBlocks,
  OPENAI_CODEX_STRUCTURED_EVENT,
  structuredCodexChunkBlocks,
} from '../src/structured-search.ts'

describe('Codex structured chunk carrier', () => {
  const block = {
    type: 'codex-web-search' as const,
    id: 'ws-live',
    status: 'searching' as const,
    action: { type: 'search' as const, query: 'layout-free search' },
  }

  it('persists metadata inside a standard assistant/chunk event', () => {
    const chunk = carryCodexStructuredBlocks({
      type: 'usage',
      usage: { inputTokens: 3, outputTokens: 2 },
    }, [block])
    const session = Session.create(SessionId('structured-search-chunk'))
    session.append('turn/start', { turn: 4 })
    session.append('step/start', { turn: 4, step: 2 })
    session.append('assistant/chunk', { turn: 4, step: 2, chunk })

    expect(KNOWN_SESSION_EVENT_TYPES.has(OPENAI_CODEX_STRUCTURED_EVENT)).toBe(false)
    expect(session.events[2]).toMatchObject({
      type: 'assistant/chunk',
      data: {
        turn: 4,
        step: 2,
        chunk: {
          type: 'usage',
          codexStructured: [expect.objectContaining({ version: 1, kind: 'web-search' })],
        },
      },
    })
    const serialized = JSON.parse(JSON.stringify(session.events[2])) as {
      data: { chunk: unknown }
    }
    expect(structuredCodexChunkBlocks(serialized.data.chunk)).toEqual([block])

    const packed = packChunkRuns(session.events)
    expect(packed[2]).toEqual(session.events[2])
    expect(packed[2]?.type).toBe('assistant/chunk')
  })

  it('does not alter the core chunk assembler result', () => {
    const assembler = new BlockAssembler()
    assembler.push(carryCodexStructuredBlocks({
      type: 'usage',
      usage: { inputTokens: 3, outputTokens: 2 },
    }, [block]))
    assembler.push({ type: 'finish', reason: { kind: 'stop' } })

    expect(assembler.blocks()).toEqual([])
    expect(assembler.usage).toEqual({ inputTokens: 3, outputTokens: 2 })
    expect(assembler.finish).toEqual({ kind: 'stop' })
  })

  it('rejects malformed metadata without affecting an ordinary chunk', () => {
    expect(() => structuredCodexChunkBlocks({
      type: 'usage',
      usage: { inputTokens: 0, outputTokens: 0 },
      codexStructured: { version: 1 },
    })).toThrow('Codex structured chunk metadata is malformed')
    expect(structuredCodexChunkBlocks({
      type: 'usage', usage: { inputTokens: 0, outputTokens: 0 },
    })).toEqual([])
  })
})
