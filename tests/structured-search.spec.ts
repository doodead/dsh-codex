import { describe, expect, it } from 'vitest'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import type { Message, StreamChunk } from '@deepseek-ai/dsh-llm'
import type { AssistantMessage, AssistantMessageEvent } from '@earendil-works/pi-ai'
import { openaiCodexProvider } from '@earendil-works/pi-ai/providers/openai-codex'
import { convertResponsesMessages } from '@earendil-works/pi-ai/api/openai-responses-shared'
import { toCodexAssistant } from '../src/codex-replay.ts'
import { toStreamChunks } from '../src/codex-stream.ts'
import {
  encodeCodexReplayMarker,
  expandCodexReplayMarkers,
  framedCodexBlock,
  foreignSourcesBlock,
  parseCodexResponseMessageBlock,
  parseCodexWebSearchBlock,
  structuredCodexBlock,
} from '../src/structured-search.ts'
import { processStructuredResponsesStream } from '../src/vendor/pi-ai/process-structured-responses.ts'

function model() {
  const provider = openaiCodexProvider()
  const value = provider.getModels()[0]
  if (value === undefined) throw new Error('Codex provider has no models')
  return value
}

function emptyAssistant(): AssistantMessage {
  const selected = model()
  return {
    role: 'assistant', content: [], api: 'openai-codex-responses',
    provider: selected.provider, model: selected.id,
    usage: {
      input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'stop', timestamp: 0,
  }
}

async function* rawEvents(): AsyncGenerator<unknown> {
  const search = {
    type: 'web_search_call', id: 'ws_1', status: 'in_progress',
  }
  yield { type: 'response.created', response: { id: 'resp_1' } }
  yield { type: 'response.output_item.added', output_index: 0, item: search }
  yield {
    type: 'response.web_search_call.searching', output_index: 0,
    item_id: 'ws_1', sequence_number: 2,
  }
  yield {
    type: 'response.output_item.done', output_index: 0,
    item: {
      ...search, status: 'completed',
      action: {
        type: 'search', query: 'dsh', queries: ['dsh', 'deepseek harness'],
        sources: [{ type: 'url', url: 'https://example.com/source' }],
      },
    },
  }
  yield {
    type: 'response.output_item.added', output_index: 1,
    item: {
      type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'inspect',
      arguments: '', status: 'in_progress',
    },
  }
  yield {
    type: 'response.function_call_arguments.delta', output_index: 1,
    item_id: 'fc_1', delta: '{"path":"README.md"}',
  }
  yield {
    type: 'response.output_item.done', output_index: 1,
    item: {
      type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'inspect',
      arguments: '{"path":"README.md"}', status: 'completed',
    },
  }
  yield {
    type: 'response.output_item.added', output_index: 2,
    item: { type: 'message', id: 'msg_1', role: 'assistant', status: 'in_progress', content: [] },
  }
  yield { type: 'response.output_text.delta', output_index: 2, content_index: 0, delta: 'Answer.' }
  yield {
    type: 'response.output_item.done', output_index: 2,
    item: {
      type: 'message', id: 'msg_1', role: 'assistant', status: 'completed', phase: 'final_answer',
      content: [{
        type: 'output_text', text: 'Answer.', annotations: [{
          type: 'url_citation', start_index: 0, end_index: 7,
          title: 'Example', url: 'https://example.com/source',
        }],
      }],
    },
  }
  yield {
    type: 'response.completed', response: {
      id: 'resp_1', status: 'completed', output: [],
      usage: {
        input_tokens: 10, output_tokens: 5, total_tokens: 15,
        input_tokens_details: { cached_tokens: 2 },
        output_tokens_details: { reasoning_tokens: 1 },
      },
    },
  }
}

describe('structured Codex search protocol', () => {
  it('rejects unknown search states and actions exhaustively', () => {
    expect(() => parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'ws', status: 'new', action: { type: 'search', query: 'x' },
    })).toThrow('unsupported Codex web search status')
    expect(() => parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'ws', status: 'completed', action: { type: 'browse' },
    })).toThrow('unsupported Codex web search action')
  })

  it('accepts an actionless live start but requires completed action data', () => {
    expect(parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'ws', status: 'in_progress',
    })).toEqual({ type: 'codex-web-search', id: 'ws', status: 'in_progress' })
    expect(() => parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'ws', status: 'completed',
    })).toThrow('Codex completed web search has no action')
    expect(() => parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'ws', status: 'in_progress', action: null,
    })).toThrow('Codex web search action is malformed')
  })

  it('validates and preserves all supported actions', () => {
    expect(parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'open', status: 'completed', action: { type: 'open_page', url: null },
    }).action).toEqual({ type: 'open_page', url: null })
    expect(parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'find', status: 'failed',
      action: { type: 'find_in_page', pattern: 'needle', url: 'https://example.com' },
    }).action).toEqual({ type: 'find_in_page', pattern: 'needle', url: 'https://example.com' })
  })

  it('rejects malformed URL citations', () => {
    expect(() => parseCodexResponseMessageBlock({
      type: 'message', id: 'msg', status: 'completed', content: [{
        type: 'output_text', text: 'x', annotations: [{
          type: 'url_citation', start_index: '0', end_index: 1, title: 'x', url: 'https://example.com',
        }],
      }],
    })).toThrow('Codex URL citation is malformed')
    expect(() => parseCodexResponseMessageBlock({
      type: 'message', id: 'msg', status: 'completed', content: [{
        type: 'output_text', text: 'x', annotations: [{
          type: 'url_citation', start_index: 0, end_index: 2, title: 'x', url: 'https://example.com',
        }],
      }],
    })).toThrow('offsets exceed output text')
  })

  it('preserves multiple native content parts, repeated citations, and offsets', () => {
    const structured = parseCodexResponseMessageBlock({
      type: 'message', id: 'multi', status: 'incomplete', phase: 'commentary',
      content: [
        {
          type: 'output_text', text: 'first', annotations: [
            { type: 'url_citation', start_index: 0, end_index: 5, title: 'One', url: 'https://one.example' },
            { type: 'url_citation', start_index: 0, end_index: 5, title: 'One again', url: 'https://one.example' },
          ],
        },
        { type: 'refusal', refusal: 'second' },
      ],
    })
    const input = [
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'firstsecond' }] },
      {
        type: 'message', role: 'assistant',
        content: [{ type: 'output_text', text: encodeCodexReplayMarker({ kind: 'response-message', block: structured }) }],
      },
    ]
    expect(expandCodexReplayMarkers(input)).toEqual([{
      type: 'message', id: 'multi', role: 'assistant', status: 'incomplete', phase: 'commentary',
      content: [
        {
          type: 'output_text', text: 'first', annotations: [
            { type: 'url_citation', start_index: 0, end_index: 5, title: 'One', url: 'https://one.example' },
            { type: 'url_citation', start_index: 0, end_index: 5, title: 'One again', url: 'https://one.example' },
          ],
        },
        { type: 'refusal', refusal: 'second' },
      ],
    }])
  })

  it('collapses each preserved search lifecycle to one native replay item', () => {
    const marker = (block: ReturnType<typeof parseCodexWebSearchBlock>) => ({
      type: 'message', role: 'assistant',
      content: [{ type: 'output_text', text: encodeCodexReplayMarker({ kind: 'web-search', block }) }],
    })
    const search = (id: string, status: 'in_progress' | 'completed' | 'failed') => parseCodexWebSearchBlock({
      type: 'web_search_call', id, status,
      action: { type: 'search', query: id, ...(status === 'completed' ? { sources: [{ type: 'url', url: `https://${id}.example` }] } : {}) },
    })
    expect(expandCodexReplayMarkers([
      marker(search('one', 'in_progress')),
      marker(search('one', 'completed')),
      marker(search('two', 'in_progress')),
      marker(search('two', 'failed')),
    ])).toEqual([
      expect.objectContaining({ type: 'web_search_call', id: 'one', status: 'completed' }),
      expect.objectContaining({ type: 'web_search_call', id: 'two', status: 'failed' }),
    ])
  })

  it.each(['SSE', 'WebSocket'])('preserves %s items, citations, durable replay, and foreign sources', async () => {
    const output = emptyAssistant()
    const pushed: Array<AssistantMessageEvent | { type: string }> = []
    await processStructuredResponsesStream(
      rawEvents(), output,
      { push: (event: AssistantMessageEvent) => { pushed.push(event) } } as never,
      model(),
    )
    pushed.push({ type: 'done', reason: 'stop', message: output } as AssistantMessageEvent)

    const eventSource = async function* (): AsyncGenerator<AssistantMessageEvent> {
      for (const event of pushed) yield event as AssistantMessageEvent
    }
    const chunks: StreamChunk[] = []
    for await (const chunk of toStreamChunks(eventSource(), model().contextWindow)) chunks.push(chunk)

    expect(chunks).toContainEqual({ type: 'block-start', index: 0, blockType: 'text' })
    const searches = chunks.flatMap(chunk => {
      if (chunk.type !== 'block-end') return []
      const structured = structuredCodexBlock(chunk.block)
      return structured?.type === 'codex-web-search' ? [structured] : []
    })
    expect(searches.map(search => search.status)).toEqual(['in_progress', 'searching', 'completed'])
    expect(chunks).toContainEqual(expect.objectContaining({
      type: 'block-end', index: 3,
      block: expect.objectContaining({ type: 'tool-call', id: expect.stringContaining('call_1'), name: 'inspect' }),
    }))
    const messageEnd = chunks.find(chunk => chunk.type === 'block-end' && chunk.index === 5)
    expect(messageEnd?.type === 'block-end' ? structuredCodexBlock(messageEnd.block) : undefined)
      .toEqual(expect.objectContaining({ type: 'codex-response-message', id: 'msg_1' }))
    expect(chunks).toContainEqual(expect.objectContaining({
      type: 'block-end', index: 6,
      block: expect.objectContaining({ type: 'text', text: expect.stringContaining('### Sources') }),
    }))

    const assembler = new BlockAssembler()
    for (const chunk of chunks) assembler.push(chunk)
    const stored = assembler.message({
      kind: 'model', provider: output.provider, model: output.model,
      replayState: assembler.replayState,
    } as never)
    const reloaded = JSON.parse(JSON.stringify(stored)) as Message
    const forked = structuredClone(reloaded)
    const replayed = toCodexAssistant(forked)
    const converted = convertResponsesMessages(model(), { messages: [replayed] }, new Set(['openai-codex']), {
      includeSystemPrompt: false,
    })
    const exact = expandCodexReplayMarkers(converted)

    expect(exact.map(item => (item as { type: string }).type)).toEqual([
      'web_search_call', 'function_call', 'message', 'function_call_output',
    ])
    expect(exact[0]).toEqual(expect.objectContaining({ id: 'ws_1', status: 'completed' }))
    expect(exact[2]).toEqual(expect.objectContaining({
      id: 'msg_1', phase: 'final_answer',
      content: [expect.objectContaining({
        annotations: [expect.objectContaining({
          type: 'url_citation', url: 'https://example.com/source',
        })],
      })],
    }))
  })

  it('deduplicates safe foreign sources and omits unsafe URLs', () => {
    const block = foreignSourcesBlock([
      parseCodexResponseMessageBlock({
        type: 'message', id: 'm', status: 'completed', content: [{
          type: 'output_text', text: 'x', annotations: [
            { type: 'url_citation', start_index: 0, end_index: 1, title: 'One', url: 'https://example.com' },
            { type: 'url_citation', start_index: 0, end_index: 1, title: 'Again', url: 'https://example.com' },
            { type: 'url_citation', start_index: 0, end_index: 1, title: 'Bad', url: 'javascript:alert(1)' },
          ],
        }],
      }),
    ])
    expect(block).toEqual({ type: 'text', text: '\n\n### Sources\n\n- [One](<https://example.com/>)' })
  })

  it('preserves plain native message identity and phase', async () => {
    const output = emptyAssistant()
    const pushed: Array<AssistantMessageEvent | { type: string }> = []
    const events = async function* (): AsyncGenerator<unknown> {
      yield {
        type: 'response.output_item.added', output_index: 0,
        item: { type: 'message', id: 'msg_plain', role: 'assistant', status: 'in_progress', content: [] },
      }
      yield { type: 'response.output_text.delta', output_index: 0, content_index: 0, delta: 'Plain.' }
      yield {
        type: 'response.output_item.done', output_index: 0,
        item: {
          type: 'message', id: 'msg_plain', role: 'assistant', status: 'completed', phase: 'commentary',
          content: [{ type: 'output_text', text: 'Plain.', annotations: [] }],
        },
      }
      yield {
        type: 'response.completed',
        response: { id: 'resp_plain', status: 'completed', output: [] },
      }
    }
    await processStructuredResponsesStream(
      events(), output,
      { push: (event: AssistantMessageEvent) => { pushed.push(event) } } as never,
      model(),
    )
    expect(output.content).toContainEqual(expect.objectContaining({
      type: 'codexResponseMessage', id: 'msg_plain', phase: 'commentary', status: 'completed',
    }))
  })

  it('keeps legacy pi-ai replay readable and degrades malformed state safely', () => {
    const legacy = {
      role: 'assistant',
      content: [{ type: 'text', text: 'legacy' }],
      source: {
        kind: 'model', provider: 'openai-codex', model: model().id,
        replayState: {
          response: {
            kind: 'pi-ai', version: 2, api: 'openai-codex-responses',
            provider: 'openai-codex', model: model().id, stopReason: 'stop',
          },
          blocks: [{ type: 'text', textSignature: 'signature' }],
        },
      },
    } as unknown as Message
    expect(toCodexAssistant(legacy).content).toEqual([
      { type: 'text', text: 'legacy', textSignature: 'signature' },
    ])

    let degraded = ''
    const malformed = structuredClone(legacy) as unknown as {
      source: { replayState: { blocks: Array<{ type: string }> } }
    }
    malformed.source.replayState.blocks[0] = { type: 'unknown' }
    expect(toCodexAssistant(malformed as unknown as Message, reason => { degraded = reason }).content)
      .toEqual([{ type: 'text', text: 'legacy' }])
    expect(degraded).toContain('unknown type')
  })

  it('drops invisible structured frames from provider-neutral replay', () => {
    const foreign = {
      role: 'assistant',
      content: [
        { type: 'text', text: 'answer' },
        framedCodexBlock(parseCodexWebSearchBlock({
          type: 'web_search_call', id: 'hidden-foreign', status: 'completed',
          action: { type: 'search', query: 'answer' },
        })),
        { type: 'text', text: '\n\n### Sources\n\n- [Example](<https://example.com/>)' },
      ],
      source: { kind: 'model', provider: 'foreign', model: 'foreign' },
    } as unknown as Message
    expect(toCodexAssistant(foreign).content).toEqual([
      { type: 'text', text: 'answer' },
      { type: 'text', text: '\n\n### Sources\n\n- [Example](<https://example.com/>)' },
    ])
  })

  it('serializes structured replay losslessly and closes failed live calls', async () => {
    const search = parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'failed', status: 'in_progress',
      action: { type: 'find_in_page', pattern: 'needle', url: 'https://example.com' },
    })
    const output = emptyAssistant()
    ;(output.content as unknown[]).push({ ...search, type: 'codexWebSearch' })
    const failure = { ...output, stopReason: 'error', errorMessage: 'search failed' } as AssistantMessage
    const events = async function* (): AsyncGenerator<AssistantMessageEvent> {
      yield { type: 'start', partial: output } as AssistantMessageEvent
      yield { type: 'codex_web_search_start', contentIndex: 0, partial: output } as unknown as AssistantMessageEvent
      yield { type: 'error', reason: 'error', error: failure } as AssistantMessageEvent
    }
    const chunks: StreamChunk[] = []
    for await (const chunk of toStreamChunks(events(), model().contextWindow)) chunks.push(chunk)
    const lifecycle = chunks.flatMap(chunk => {
      if (chunk.type !== 'block-end') return []
      const structured = structuredCodexBlock(chunk.block)
      return structured?.type === 'codex-web-search' ? [structured] : []
    })
    expect(lifecycle).toEqual([search, { ...search, status: 'failed' }])
    expect(chunks.at(-1)).toEqual(expect.objectContaining({
      type: 'finish', reason: expect.objectContaining({ kind: 'error' }),
      replayState: expect.objectContaining({
        blocks: [
          expect.objectContaining({ type: 'codex-web-search' }),
          expect.objectContaining({ type: 'codex-web-search' }),
        ],
      }),
    }))

    const roundTrip = JSON.parse(JSON.stringify(framedCodexBlock(search))) as unknown
    expect(structuredCodexBlock(roundTrip as Message['content'][number])).toEqual(search)
  })

  it('keeps interrupted searches open for durable interrupted rendering', async () => {
    const search = parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'aborted', status: 'in_progress',
      action: { type: 'search', query: 'unfinished' },
    })
    const output = emptyAssistant()
    ;(output.content as unknown[]).push({ ...search, type: 'codexWebSearch' })
    const failure = { ...output, stopReason: 'aborted', errorMessage: 'cancelled' } as AssistantMessage
    const events = async function* (): AsyncGenerator<AssistantMessageEvent> {
      yield { type: 'start', partial: output } as AssistantMessageEvent
      yield { type: 'codex_web_search_start', contentIndex: 0, partial: output } as unknown as AssistantMessageEvent
      yield { type: 'error', reason: 'aborted', error: failure } as AssistantMessageEvent
    }
    const chunks: StreamChunk[] = []
    for await (const chunk of toStreamChunks(events(), model().contextWindow)) chunks.push(chunk)
    const lifecycle = chunks.flatMap(chunk => {
      if (chunk.type !== 'block-end') return []
      const structured = structuredCodexBlock(chunk.block)
      return structured?.type === 'codex-web-search' ? [structured.status] : []
    })
    expect(lifecycle).toEqual(['in_progress'])
    expect(chunks.at(-1)).toEqual(expect.objectContaining({
      type: 'finish', reason: expect.objectContaining({ kind: 'aborted' }),
      replayState: expect.objectContaining({
        blocks: [expect.objectContaining({ type: 'codex-web-search' })],
      }),
    }))
  })

  it('keeps structured framing invisible to the stock assistant renderer', () => {
    expect(framedCodexBlock(parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'hidden', status: 'completed',
      action: { type: 'search', query: 'hidden' },
    }))).toMatchObject({ type: 'text', text: '' })
  })
})
