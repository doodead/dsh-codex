import { describe, expect, it } from 'vitest'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import type { Message, StreamChunk } from '@deepseek-ai/dsh-llm'
import type { AssistantMessage, AssistantMessageEvent } from '@earendil-works/pi-ai'
import { openaiCodexProvider } from '@earendil-works/pi-ai/providers/openai-codex'
import { convertResponsesMessages } from '@earendil-works/pi-ai/api/openai-responses-shared'
import { toCodexAssistant, toCodexReplayState } from '../src/codex-replay.ts'
import { toStreamChunks } from '../src/codex-stream.ts'
import {
  encodeCodexReplayMarker,
  expandCodexReplayMarkers,
  framedCodexBlock,
  foreignSourcesBlock,
  parseCodexResponseMessageBlock,
  parseCodexWebSearchBlock,
  structuredCodexBlock,
  structuredCodexChunkBlocks,
  structuredCodexReplayBlocks,
} from '../src/structured-search.ts'
import type { CodexStructuredBlock } from '../src/structured-search.ts'
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

  it('hides empty native reasoning without losing exact replay', async () => {
    const reasoningItem = {
      type: 'reasoning', id: 'rs_empty', summary: [],
      encrypted_content: 'encrypted-empty-reasoning',
    }
    const { encrypted_content: _terminalOnly, ...doneReasoningItem } = reasoningItem
    const output = emptyAssistant()
    const pushed: AssistantMessageEvent[] = []
    const raw = async function* (): AsyncGenerator<unknown> {
      yield {
        type: 'response.output_item.added', output_index: 0,
        item: { type: 'reasoning', id: 'rs_empty', summary: [], status: 'in_progress' },
      }
      yield {
        type: 'response.output_item.done', output_index: 0,
        item: doneReasoningItem,
      }
      yield {
        type: 'response.completed',
        response: { id: 'resp_empty', status: 'completed', output: [reasoningItem] },
      }
    }
    await processStructuredResponsesStream(
      raw(), output,
      { push: (event: AssistantMessageEvent) => { pushed.push(event) } } as never,
      model(),
    )
    pushed.push({ type: 'done', reason: 'stop', message: output })
    const events = async function* (): AsyncGenerator<AssistantMessageEvent> {
      yield* pushed
    }

    const chunks: StreamChunk[] = []
    for await (const chunk of toStreamChunks(events(), model().contextWindow)) chunks.push(chunk)

    expect(chunks.some(chunk => chunk.type === 'block-start' && chunk.blockType === 'reasoning')).toBe(false)
    expect(chunks.some(chunk => chunk.type === 'reasoning-delta')).toBe(false)
    expect(chunks.some(chunk => chunk.type === 'block-end')).toBe(false)

    const assembler = new BlockAssembler()
    for (const chunk of chunks) assembler.push(chunk)
    expect(assembler.replayState).toBeDefined()
    const stored = assembler.message({
      kind: 'model', provider: output.provider, model: output.model,
      replayState: assembler.replayState,
    } as never)
    expect(stored.content).toEqual([])
    const reloaded = JSON.parse(JSON.stringify(stored)) as Message
    const replayed = toCodexAssistant(reloaded)
    expect(replayed.content).toEqual([{
      type: 'thinking', thinking: '',
      thinkingSignature: JSON.stringify(reasoningItem),
    }])
    expect(convertResponsesMessages(model(), { messages: [replayed] }, new Set(['openai-codex']), {
      includeSystemPrompt: false,
    })).toEqual([reasoningItem])
    expect(toCodexAssistant({
      ...stored,
      source: { kind: 'model', provider: 'foreign', model: 'foreign' },
    } as Message).content).toEqual([])
  })

  it('keeps hidden reasoning ordered around structured search replay', () => {
    const before = {
      type: 'reasoning', id: 'rs_before_search', summary: [],
      encrypted_content: 'encrypted-before-search',
    }
    const after = {
      type: 'reasoning', id: 'rs_after_search', summary: [],
      encrypted_content: 'encrypted-after-search',
    }
    const searches = [
      parseCodexWebSearchBlock({
        type: 'web_search_call', id: 'ws_interleaved', status: 'in_progress',
      }),
      parseCodexWebSearchBlock({
        type: 'web_search_call', id: 'ws_interleaved', status: 'searching',
      }),
      parseCodexWebSearchBlock({
        type: 'web_search_call', id: 'ws_interleaved', status: 'completed',
        action: { type: 'search', query: 'trace-shaped search' },
      }),
    ]
    const output = emptyAssistant()
    ;(output.content as unknown[]).push(
      { type: 'thinking', thinking: '', thinkingSignature: JSON.stringify(before) },
      ...searches.map(search => ({ ...search, type: 'codexWebSearch' })),
      { type: 'thinking', thinking: '', thinkingSignature: JSON.stringify(after) },
    )
    const replayState = toCodexReplayState(output, false)
    const stored = {
      role: 'assistant',
      content: [],
      source: {
        kind: 'model', provider: output.provider, model: output.model, replayState,
      },
    } as unknown as Message

    expect(replayState).toEqual(expect.objectContaining({
      response: expect.objectContaining({
        version: 2,
        blockTypes: [],
        detached: [
          { position: 0, item: { type: 'reasoning', thinkingSignature: JSON.stringify(before) } },
          ...searches.map(block => ({
            position: 0, item: { type: 'codex-web-search' as const, block },
          })),
          { position: 0, item: { type: 'reasoning', thinkingSignature: JSON.stringify(after) } },
        ],
      }),
      blocks: [],
    }))
    const replayed = toCodexAssistant(JSON.parse(JSON.stringify(stored)) as Message)
    const exact = expandCodexReplayMarkers(convertResponsesMessages(
      model(), { messages: [replayed] }, new Set(['openai-codex']),
      { includeSystemPrompt: false },
    ))
    expect(exact).toEqual([
      before,
      expect.objectContaining({
        type: 'web_search_call', id: 'ws_interleaved', status: 'completed',
      }),
      after,
    ])
  })

  it('continues streaming non-empty native reasoning', async () => {
    const output = emptyAssistant()
    output.content.push({ type: 'thinking', thinking: 'visible thought' })
    const events = async function* (): AsyncGenerator<AssistantMessageEvent> {
      yield { type: 'start', partial: output }
      yield { type: 'thinking_start', contentIndex: 0, partial: output }
      yield { type: 'thinking_delta', contentIndex: 0, delta: '\n\n', partial: output }
      yield { type: 'thinking_delta', contentIndex: 0, delta: 'visible thought', partial: output }
      yield { type: 'thinking_end', contentIndex: 0, content: '\n\nvisible thought', partial: output }
      yield { type: 'done', reason: 'stop', message: output }
    }

    const chunks: StreamChunk[] = []
    for await (const chunk of toStreamChunks(events(), model().contextWindow)) chunks.push(chunk)

    expect(chunks).toContainEqual({ type: 'block-start', index: 0, blockType: 'reasoning' })
    expect(chunks).toContainEqual({ type: 'reasoning-delta', index: 0, text: '\n\nvisible thought' })
    expect(chunks).toContainEqual({
      type: 'block-end', index: 0,
      block: { type: 'reasoning', text: '\n\nvisible thought' },
    })
  })

  it('streams search metadata on a real block without creating empty content', async () => {
    const started = parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'ws_live', status: 'in_progress',
    })
    const searching = { ...started, status: 'searching' as const }
    const completed = parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'ws_live', status: 'completed',
      action: { type: 'search', query: 'metadata carrier' },
    })
    const output = emptyAssistant()
    ;(output.content as unknown[]).push(
      { type: 'thinking', thinking: 'visible thought' },
      { ...started, type: 'codexWebSearch' },
      { type: 'text', text: 'answer' },
    )
    const events = async function* (): AsyncGenerator<AssistantMessageEvent> {
      yield { type: 'start', partial: output }
      yield { type: 'thinking_start', contentIndex: 0, partial: output }
      yield { type: 'thinking_delta', contentIndex: 0, delta: 'visible thought', partial: output }
      yield { type: 'thinking_end', contentIndex: 0, content: 'visible thought', partial: output }
      yield { type: 'codex_web_search_start', contentIndex: 1, partial: output } as unknown as AssistantMessageEvent
      yield {
        type: 'codex_web_search_update', contentIndex: 1,
        block: { ...searching, type: 'codexWebSearch' }, partial: output,
      } as unknown as AssistantMessageEvent
      ;(output.content as unknown[])[1] = { ...completed, type: 'codexWebSearch' }
      yield {
        type: 'codex_web_search_end', contentIndex: 1,
        block: { ...completed, type: 'codexWebSearch' }, partial: output,
      } as unknown as AssistantMessageEvent
      yield { type: 'text_start', contentIndex: 2, partial: output }
      yield { type: 'text_delta', contentIndex: 2, delta: 'answer', partial: output }
      yield { type: 'text_end', contentIndex: 2, content: 'answer', partial: output }
      yield { type: 'done', reason: 'stop', message: output }
    }

    const chunks: StreamChunk[] = []
    for await (const chunk of toStreamChunks(events(), model().contextWindow)) chunks.push(chunk)

    const carriers = chunks.filter(chunk => structuredCodexChunkBlocks(chunk).length > 0)
    expect(carriers).toHaveLength(3)
    expect(carriers.every(chunk => chunk.type === 'reasoning-delta'
      && chunk.index === 0 && chunk.text === '')).toBe(true)
    expect(carriers.flatMap(chunk => structuredCodexChunkBlocks(chunk))).toEqual([
      started, searching, completed,
    ])
    expect(chunks.findIndex(chunk => chunk.type === 'block-end' && chunk.index === 0))
      .toBeGreaterThan(chunks.lastIndexOf(carriers[2] as StreamChunk))

    const assembler = new BlockAssembler()
    for (const chunk of chunks) assembler.push(chunk)
    expect(assembler.blocks()).toEqual([
      { type: 'reasoning', text: 'visible thought' },
      { type: 'text', text: 'answer' },
    ])
  })

  it('anchors hidden reasoning before the next retained block', async () => {
    const reasoningItem = {
      type: 'reasoning', id: 'rs_before_text', summary: [], encrypted_content: 'encrypted-before-text',
    }
    const output = emptyAssistant()
    output.content.push(
      { type: 'thinking', thinking: '', thinkingSignature: JSON.stringify(reasoningItem) },
      { type: 'text', text: 'answer', textSignature: 'text-signature' },
    )
    const events = async function* (): AsyncGenerator<AssistantMessageEvent> {
      yield { type: 'start', partial: output }
      yield { type: 'thinking_start', contentIndex: 0, partial: output }
      yield { type: 'thinking_end', contentIndex: 0, content: '', partial: output }
      yield { type: 'text_start', contentIndex: 1, partial: output }
      yield { type: 'text_delta', contentIndex: 1, delta: 'answer', partial: output }
      yield { type: 'text_end', contentIndex: 1, content: 'answer', partial: output }
      yield { type: 'done', reason: 'stop', message: output }
    }

    const assembler = new BlockAssembler()
    for await (const chunk of toStreamChunks(events(), model().contextWindow)) assembler.push(chunk)
    const stored = assembler.message({
      kind: 'model', provider: output.provider, model: output.model,
      replayState: assembler.replayState,
    } as never)

    expect(stored.content).toEqual([{ type: 'text', text: 'answer' }])
    expect(assembler.replayState).toEqual(expect.objectContaining({
      response: expect.objectContaining({
        detached: [{
          position: 0,
          item: { type: 'reasoning', thinkingSignature: JSON.stringify(reasoningItem) },
        }],
      }),
      blocks: [{ type: 'text', textSignature: 'text-signature', ordinal: 0 }],
    }))
    expect(toCodexAssistant(stored).content).toEqual([
      { type: 'thinking', thinking: '', thinkingSignature: JSON.stringify(reasoningItem) },
      { type: 'text', text: 'answer', textSignature: 'text-signature' },
    ])
  })

  it('prunes hidden reasoning with a dropped max-token tool call', async () => {
    const output = emptyAssistant()
    output.stopReason = 'length'
    output.content.push(
      { type: 'thinking', thinking: '', thinkingSignature: '{"type":"reasoning","id":"rs_tool"}' },
      { type: 'toolCall', id: 'call_1|fc_1', name: 'inspect', arguments: { path: 'README.md' } },
    )
    const events = async function* (): AsyncGenerator<AssistantMessageEvent> {
      yield { type: 'start', partial: output }
      yield { type: 'thinking_start', contentIndex: 0, partial: output }
      yield { type: 'thinking_end', contentIndex: 0, content: '', partial: output }
      yield { type: 'toolcall_start', contentIndex: 1, partial: output }
      yield { type: 'toolcall_delta', contentIndex: 1, delta: '{"path":"README.md"}', partial: output }
      yield { type: 'toolcall_end', contentIndex: 1, toolCall: output.content[1] as never, partial: output }
      yield { type: 'done', reason: 'length', message: output }
    }

    const assembler = new BlockAssembler()
    for await (const chunk of toStreamChunks(events(), model().contextWindow)) assembler.push(chunk)

    expect(assembler.blocks()).toEqual([])
    expect(assembler.replayState).toEqual(expect.objectContaining({ blocks: [] }))
    const stored = assembler.message({
      kind: 'model', provider: output.provider, model: output.model,
      replayState: assembler.replayState,
    } as never)
    expect(toCodexAssistant(stored).content).toEqual([])
  })

  it('keeps detached search replay while pruning a later max-token tool call', () => {
    const search = parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'ws_before_pruned_tool', status: 'completed',
      action: { type: 'search', query: 'retained search' },
    })
    const output = emptyAssistant()
    output.stopReason = 'length'
    ;(output.content as unknown[]).push(
      { ...search, type: 'codexWebSearch' },
      {
        type: 'thinking', thinking: '',
        thinkingSignature: '{"type":"reasoning","id":"rs_pruned_tool"}',
      },
      { type: 'toolCall', id: 'call_pruned|fc_pruned', name: 'inspect', arguments: {} },
    )
    const assembler = new BlockAssembler()
    assembler.push({ type: 'block-start', index: 2, blockType: 'tool-call' })
    assembler.push({
      type: 'block-end', index: 2,
      block: { type: 'tool-call', id: 'call_pruned|fc_pruned' as never, name: 'inspect', arguments: '{}' },
    })
    assembler.push({
      type: 'finish', reason: { kind: 'max-tokens' },
      replayState: toCodexReplayState(output, false),
    })
    const stored = assembler.message({
      kind: 'model', provider: output.provider, model: output.model,
      replayState: assembler.replayState,
    } as never)
    expect(stored.content).toEqual([])
    const exact = expandCodexReplayMarkers(convertResponsesMessages(
      model(), { messages: [toCodexAssistant(stored)] }, new Set(['openai-codex']),
      { includeSystemPrompt: false },
    ))
    expect(exact).toEqual([
      expect.objectContaining({
        type: 'web_search_call', id: 'ws_before_pruned_tool', status: 'completed',
      }),
    ])
  })

  it('closes interrupted empty reasoning invisibly with aligned replay', async () => {
    const output = emptyAssistant()
    output.content.push({ type: 'thinking', thinking: '' })
    const failure = { ...output, stopReason: 'aborted', errorMessage: 'cancelled' } as AssistantMessage
    const events = async function* (): AsyncGenerator<AssistantMessageEvent> {
      yield { type: 'start', partial: output }
      yield { type: 'thinking_start', contentIndex: 0, partial: output }
      yield { type: 'error', reason: 'aborted', error: failure }
    }

    const chunks: StreamChunk[] = []
    for await (const chunk of toStreamChunks(events(), model().contextWindow)) chunks.push(chunk)
    const assembler = new BlockAssembler()
    for (const chunk of chunks) assembler.push(chunk)

    expect(assembler.blocks()).toEqual([])
    expect(assembler.replayState).toEqual(expect.objectContaining({
      response: expect.objectContaining({
        detached: [{ position: 0, item: { type: 'reasoning' } }],
      }),
      blocks: [],
    }))
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
    const structured = chunks.flatMap(chunk => structuredCodexChunkBlocks(chunk))

    expect(chunks.filter(chunk => chunk.type === 'usage')).toHaveLength(1)
    expect(chunks).not.toContainEqual({ type: 'block-start', index: 0, blockType: 'text' })
    const searches = structured.filter(
      (block): block is Extract<CodexStructuredBlock, { type: 'codex-web-search' }> =>
        block.type === 'codex-web-search',
    )
    expect(searches.map(search => search.status)).toEqual(['in_progress', 'searching', 'completed'])
    expect(chunks).toContainEqual(expect.objectContaining({
      type: 'block-end', index: 3,
      block: expect.objectContaining({ type: 'tool-call', id: expect.stringContaining('call_1'), name: 'inspect' }),
    }))
    const messageEnd = structured.find(block => block.type === 'codex-response-message')
    expect(messageEnd)
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
    expect(stored.content).toEqual([
      expect.objectContaining({ type: 'tool-call' }),
      { type: 'text', text: 'Answer.' },
      expect.objectContaining({ type: 'text', text: expect.stringContaining('### Sources') }),
    ])
    expect(stored.content.some(block => block.type === 'text' && block.text === '')).toBe(false)
    expect(structuredCodexReplayBlocks(assembler.replayState)).toEqual([
      ...searches,
      expect.objectContaining({ type: 'codex-response-message', id: 'msg_1' }),
    ])
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

    const legacyEmptyReasoning = {
      role: 'assistant',
      content: [{ type: 'reasoning', text: '' }],
      source: {
        kind: 'model', provider: 'openai-codex', model: model().id,
        replayState: {
          response: {
            kind: 'dsh-codex-pi-ai', version: 1, api: 'openai-codex-responses',
            provider: 'openai-codex', model: model().id, stopReason: 'stop',
          },
          blocks: [{
            type: 'reasoning',
            thinkingSignature: '{"type":"reasoning","id":"legacy-empty","summary":[]}',
          }],
        },
      },
    } as unknown as Message
    expect(toCodexAssistant(legacyEmptyReasoning).content).toEqual([{
      type: 'thinking', thinking: '',
      thinkingSignature: '{"type":"reasoning","id":"legacy-empty","summary":[]}',
    }])

    const legacySearch = parseCodexWebSearchBlock({
      type: 'web_search_call', id: 'legacy-search', status: 'completed',
      action: { type: 'search', query: 'legacy' },
    })
    const legacyStructured = {
      role: 'assistant',
      content: [framedCodexBlock(legacySearch)],
      source: {
        kind: 'model', provider: 'openai-codex', model: model().id,
        replayState: {
          response: {
            kind: 'dsh-codex-pi-ai', version: 1, api: 'openai-codex-responses',
            provider: 'openai-codex', model: model().id, stopReason: 'stop',
          },
          blocks: [{ type: 'codex-web-search', block: legacySearch }],
        },
      },
    } as unknown as Message
    expect(structuredCodexReplayBlocks(
      legacyStructured.source.kind === 'model' ? legacyStructured.source.replayState : undefined,
    )).toEqual([legacySearch])
    const exactLegacy = expandCodexReplayMarkers(convertResponsesMessages(
      model(), { messages: [toCodexAssistant(legacyStructured)] }, new Set(['openai-codex']),
      { includeSystemPrompt: false },
    ))
    expect(exactLegacy).toEqual([expect.objectContaining({
      type: 'web_search_call', id: 'legacy-search', status: 'completed',
    })])
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
    const lifecycle = chunks.flatMap(chunk => structuredCodexChunkBlocks(chunk))
    expect(lifecycle).toEqual([search, { ...search, status: 'failed' }])
    expect(chunks.at(-1)).toEqual(expect.objectContaining({
      type: 'finish', reason: expect.objectContaining({ kind: 'error' }),
      replayState: expect.objectContaining({
        response: expect.objectContaining({
          detached: [
            expect.objectContaining({ item: expect.objectContaining({ type: 'codex-web-search' }) }),
            expect.objectContaining({ item: expect.objectContaining({ type: 'codex-web-search' }) }),
          ],
        }),
        blocks: [],
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
    const lifecycle = chunks.flatMap(chunk => structuredCodexChunkBlocks(chunk))
    expect(lifecycle.map(block => block.type === 'codex-web-search' ? block.status : undefined))
      .toEqual(['in_progress'])
    expect(chunks.at(-1)).toEqual(expect.objectContaining({
      type: 'finish', reason: expect.objectContaining({ kind: 'aborted' }),
      replayState: expect.objectContaining({
        response: expect.objectContaining({
          detached: [expect.objectContaining({
            item: expect.objectContaining({ type: 'codex-web-search' }),
          })],
        }),
        blocks: [],
      }),
    }))
  })

})
