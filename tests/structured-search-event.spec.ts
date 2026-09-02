import { describe, expect, it, vi } from 'vitest'
import type { Context } from '@deepseek-ai/cordis'
import { KNOWN_SESSION_EVENT_TYPES, Session, SessionId } from '@deepseek-ai/dsh-session'
import {
  installOpenAICodexStructuredEvent,
  recordOpenAICodexStructuredBlock,
} from '../src/structured-search-event.ts'
import {
  OPENAI_CODEX_STRUCTURED_EVENT,
  parseCodexStructuredFrame,
} from '../src/structured-search.ts'

describe('Codex structured session event bridge', () => {
  it('registers its durable event type and records against the open step', () => {
    const session = Session.create(SessionId('structured-search-event'))
    session.append('turn/start', { turn: 4 })
    session.append('step/start', { turn: 4, step: 2 })
    const ctx = {
      get: (name: string) => name === 'agents'
        ? { currentInitiator: () => ({ session }) }
        : undefined,
    } as unknown as Context
    const block = {
      type: 'codex-web-search' as const,
      id: 'ws-live',
      status: 'searching' as const,
      action: { type: 'search' as const, query: 'layout-free search' },
    }

    installOpenAICodexStructuredEvent()
    recordOpenAICodexStructuredBlock(ctx, block)

    expect(KNOWN_SESSION_EVENT_TYPES.has(OPENAI_CODEX_STRUCTURED_EVENT)).toBe(true)
    expect(session.events).toHaveLength(3)
    expect(session.events[2]).toMatchObject({
      type: OPENAI_CODEX_STRUCTURED_EVENT,
      data: {
        turn: 4,
        step: 2,
        frame: expect.objectContaining({ version: 1, kind: 'web-search' }),
      },
    })
    const recorded = session.events[2]
    if (recorded?.type !== OPENAI_CODEX_STRUCTURED_EVENT) throw new Error('structured event was not recorded')
    expect(parseCodexStructuredFrame(recorded.data.frame).block).toEqual(block)
  })

  it('does not attach records outside an initiating open step', () => {
    const append = vi.fn()
    const session = {
      events: [
        { type: 'step/start', data: { turn: 1, step: 1 } },
        { type: 'step/end', data: { turn: 1, step: 1 } },
      ],
      append,
    }
    const block = {
      type: 'codex-web-search' as const,
      id: 'ws-maintenance',
      status: 'completed' as const,
    }
    const withClosedStep = {
      get: () => ({ currentInitiator: () => ({ session }) }),
    } as unknown as Context
    const withoutInitiator = {
      get: () => ({ currentInitiator: () => undefined }),
    } as unknown as Context

    recordOpenAICodexStructuredBlock(withClosedStep, block)
    recordOpenAICodexStructuredBlock(withoutInitiator, block)

    expect(append).not.toHaveBeenCalled()
  })
})
