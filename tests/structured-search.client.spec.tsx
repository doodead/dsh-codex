// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { ComponentType } from 'react'
import type { ChatConversationViewNode, ConversationMatch } from '@deepseek-ai/dsh-client-runtime/client'
import {
  CodexWebSearchNode,
  codexWebSearchDefinition,
} from '../src/client/CodexWebSearchNode.tsx'
import { en, zh } from '../src/client/locales.ts'
import type { OpenAICodexSettingsKey } from '../src/client/locales.ts'
import {
  carryCodexStructuredBlocks,
  framedCodexBlock,
  structuredCodexChunkBlocks,
} from '../src/structured-search.ts'
import { inject as clientInject } from '../src/client/index.tsx'

function t(key: OpenAICodexSettingsKey, values?: Record<string, unknown>): string {
  return Object.entries(values ?? {}).reduce(
    (text, [name, value]) => text.replaceAll(`{${name}}`, String(value)),
    en[key],
  )
}

function match(event: unknown, role: 'start' | 'update', location: ConversationMatch['location']): ConversationMatch {
  return { event, view: undefined, role, location } as ConversationMatch
}

function event(type: string, data: unknown, seq: number): unknown {
  return { type, data, seq, time: seq * 10 }
}

afterEach(cleanup)

describe('Codex hosted-search conversation row', () => {
  it('publishes live, settled, and interrupted lifecycle states', () => {
    const openLocation: ConversationMatch['location'] = {
      kind: 'step',
      turn: { turn: 1, start: undefined, end: undefined, status: 'open', steps: [], data: { get: () => undefined } },
      step: { turn: 1, step: 2, start: undefined, end: undefined, status: 'open', data: { get: () => undefined } },
    }
    const start = match(event('step/start', { turn: 1, step: 2 }, 1), 'start', openLocation)
    const context = {
      key: 'codex-web-search:1:2', kind: 'codex-web-search', id: '1:2',
      matches: [start], start, state: undefined, current: new Map(),
    }
    const state = codexWebSearchDefinition.start(context, start, { previous: () => undefined })
    const began = match(event('assistant/chunk', {
      turn: 1, step: 2,
      chunk: carryCodexStructuredBlocks({
        type: 'usage', usage: { inputTokens: 0, outputTokens: 0 },
      }, [{
        type: 'codex-web-search', id: 'ws-1', status: 'in_progress',
        action: { type: 'search', query: 'dsh', queries: ['dsh'] },
      }]),
    }, 2), 'update', openLocation)
    const runningState = codexWebSearchDefinition.update(
      { ...context, state, matches: [start, began] },
      began,
    )
    const running = codexWebSearchDefinition.buildViewNode?.({
      ...context, state: runningState, matches: [start, began],
    }) as ChatConversationViewNode
    expect(running.data).toMatchObject({ status: 'running', calls: [{ status: 'in_progress' }] })

    const completed = match(event('assistant/message', {
      turn: 1, step: 2,
      message: {
        content: [],
        source: {
          kind: 'model', provider: 'openai-codex', model: 'gpt-test',
          replayState: {
            response: {
              kind: 'dsh-codex-pi-ai', version: 2, api: 'openai-codex-responses',
              provider: 'openai-codex', model: 'gpt-test', stopReason: 'stop', blockTypes: [],
              detached: [
                {
                  position: 0,
                  item: {
                    type: 'codex-web-search',
                    block: {
                      type: 'codex-web-search', id: 'ws-1', status: 'completed',
                      action: { type: 'search', query: 'dsh', queries: ['dsh'] },
                    },
                  },
                },
                {
                  position: 0,
                  item: {
                    type: 'codex-response-message',
                    block: {
                      type: 'codex-response-message', id: 'msg-1', status: 'completed',
                      content: [{
                        type: 'output_text', text: 'answer', annotations: [{
                          type: 'url_citation', startIndex: 0, endIndex: 6,
                          title: 'Example', url: 'https://example.com',
                        }],
                      }],
                    },
                  },
                },
              ],
            },
            blocks: [],
          },
        },
      },
    }, 3), 'update', openLocation)
    const settledState = codexWebSearchDefinition.update(
      { ...context, state: runningState, matches: [start, began, completed] },
      completed,
    )
    const settled = codexWebSearchDefinition.buildViewNode?.({
      ...context, state: settledState, matches: [start, began, completed],
    }) as ChatConversationViewNode
    expect(settled.data).toMatchObject({
      status: 'settled',
      calls: [{ key: 'ws-1', status: 'completed' }],
      citations: [{ title: 'Example', url: 'https://example.com' }],
    })

    const closedLocation: ConversationMatch['location'] = {
      ...openLocation,
      step: { ...openLocation.step, status: 'closed' },
    }
    const interrupted = codexWebSearchDefinition.buildViewNode?.({
      ...context,
      state: runningState,
      start: { ...start, location: closedLocation },
      matches: [{ ...began, location: closedLocation }],
    }) as ChatConversationViewNode
    expect(interrupted.data).toMatchObject({ status: 'interrupted' })

    const persistedOpen = match(event('assistant/message', {
      turn: 1, step: 2,
      message: { content: [framedCodexBlock({
        type: 'codex-web-search', id: 'ws-1', status: 'searching',
        action: { type: 'search', query: 'dsh' },
      })] },
    }, 4), 'update', closedLocation)
    const persistedState = codexWebSearchDefinition.update(
      { ...context, state: runningState, matches: [start, persistedOpen] },
      persistedOpen,
    )
    const persistedInterrupted = codexWebSearchDefinition.buildViewNode?.({
      ...context,
      state: persistedState,
      start: { ...start, location: closedLocation },
      matches: [persistedOpen],
    }) as ChatConversationViewNode
    expect(persistedInterrupted.data).toMatchObject({ status: 'interrupted' })
  })

  it('renders only HTTP(S) sources as links', () => {
    const node = {
      key: 'search', kind: 'codex-web-search', id: '1:2', target: 'chat', anchorSeq: 1,
      location: { kind: 'unresolved' }, visibility: 'visible',
      data: {
        status: 'settled',
        calls: [{
          key: 'ws-1', status: 'completed',
          action: {
            type: 'search', query: 'query',
            sources: [
              { type: 'url', url: 'https://example.com/source' },
              { type: 'url', url: 'javascript:alert(1)' },
            ],
          },
        }],
        citations: [
          { type: 'url_citation', startIndex: 0, endIndex: 1, title: 'Citation', url: 'http://example.org' },
          { type: 'url_citation', startIndex: 0, endIndex: 1, title: 'Unsafe', url: 'file:///etc/passwd' },
        ],
      },
    }
    const TestNode = CodexWebSearchNode as ComponentType<{ node: unknown; t: typeof t }>
    render(<TestNode node={node} t={t} />)
    expect(screen.getByText('Searched the web (1)')).toBeTruthy()
    expect(screen.getAllByRole('link')).toHaveLength(2)
    expect(screen.getByRole('link', { name: 'https://example.com/source' }).getAttribute('href'))
      .toBe('https://example.com/source')
    expect(screen.getByRole('link', { name: 'Citation' }).getAttribute('href'))
      .toBe('http://example.org')
    expect(screen.getByText('javascript:alert(1)').closest('a')).toBeNull()
    expect(screen.getByText('Unsafe').closest('a')).toBeNull()
  })

  it('summarizes a failed final call', () => {
    const TestNode = CodexWebSearchNode as ComponentType<{ node: unknown; t: typeof t }>
    render(<TestNode node={{
      key: 'failed', kind: 'codex-web-search', id: '1:2', target: 'chat', anchorSeq: 1,
      location: { kind: 'unresolved' }, visibility: 'visible',
      data: {
        status: 'settled', citations: [],
        calls: [{
          key: 'ws-failed', status: 'failed',
          action: { type: 'open_page', url: 'https://example.com' },
        }],
      },
    }} t={t} />)
    expect(screen.getByText('Web search failed (1)')).toBeTruthy()
  })

  it('renders open-page and find-in-page action URLs safely', () => {
    const TestNode = CodexWebSearchNode as ComponentType<{ node: unknown; t: typeof t }>
    render(<TestNode node={{
      key: 'actions', kind: 'codex-web-search', id: '1:2', target: 'chat', anchorSeq: 1,
      location: { kind: 'unresolved' }, visibility: 'visible',
      data: {
        status: 'settled', citations: [],
        calls: [
          {
            key: 'open', status: 'completed',
            action: { type: 'open_page', url: 'https://example.com/open' },
          },
          {
            key: 'find', status: 'completed',
            action: { type: 'find_in_page', pattern: 'needle', url: 'javascript:alert(1)' },
          },
        ],
      },
    }} t={t} />)
    expect(screen.getByRole('link', { name: 'https://example.com/open' })).toBeTruthy()
    expect(screen.getByText('javascript:alert(1)').closest('a')).toBeNull()
    expect(screen.getByText('Found “needle” in a page')).toBeTruthy()
  })

  it('ships both renderer locales', () => {
    expect(en.webSearchSearching).toBe('Searching the web…')
    expect(zh.webSearchSearching).toBe('正在搜索网页…')
  })

  it('keeps conversation rendering an optional client capability', () => {
    expect(clientInject).not.toContain('conversationEvents')
  })

  it('uses standard chunks for layout-free live framing', () => {
    const chunk = carryCodexStructuredBlocks({
      type: 'usage', usage: { inputTokens: 0, outputTokens: 0 },
    }, [{
      type: 'codex-web-search', id: 'hidden', status: 'completed',
      action: { type: 'search', query: 'hidden' },
    }])
    expect(chunk).toMatchObject({ type: 'usage' })
    expect(structuredCodexChunkBlocks(chunk)).toEqual([
      expect.objectContaining({ type: 'codex-web-search', id: 'hidden' }),
    ])
  })
})
