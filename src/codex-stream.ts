/**
 * pi-ai assistant event translation into the Harness streaming protocol.
 *
 * pi-ai tool-call arguments are parsed objects while the Harness keeps their
 * raw JSON representation. pi-ai also reports failures as terminal stream
 * events, which this module maps into Harness finish chunks.
 *
 * @module dsh-llm-pi-ai/stream
 */

import { CallId, CONTEXT_WINDOW_EXCEEDED_CODE, EMPTY_RESPONSE_CODE, isContextWindowExceededError, isQuotaExceededError, LlmError, QUOTA_EXCEEDED_CODE } from '@deepseek-ai/dsh-llm'
import type { FinishReason, StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm'
import { isContextOverflow } from '@earendil-works/pi-ai'
import type { AssistantMessage, AssistantMessageEvent, Usage as PiUsage } from '@earendil-works/pi-ai'
import { toCodexReplayState } from './codex-replay.ts'
import { foreignSourcesBlock, framedCodexBlock } from './structured-search.ts'
import type {
  CodexPiResponseMessageContent,
  CodexPiWebSearchContent,
  CodexStructuredBlock,
  CodexWebSearchBlock,
} from './structured-search.ts'
import type { CodexStructuredAssistantEvent } from './vendor/pi-ai/process-structured-responses.ts'

/**
 * Map pi-ai usage (reasoning folded into output by pi-ai).
 * @param usage - cumulative usage from the terminal pi-ai event.
 * @returns harness counts; cache fields appear only when non-zero (pi-ai reports zeros, not absence).
 */
export function mapUsage(usage: PiUsage): TokenUsage {
  return {
    inputTokens: usage.input,
    outputTokens: usage.output,
    ...usage.cacheRead > 0 ? { cacheReadTokens: usage.cacheRead } : {},
    ...usage.cacheWrite > 0 ? { cacheWriteTokens: usage.cacheWrite } : {},
  }
}

function structuredBlock(block: unknown): CodexStructuredBlock | undefined {
  const candidate = block as { type?: unknown }
  switch (candidate.type) {
    case 'codexWebSearch': {
      const search = block as CodexPiWebSearchContent
      return { ...search, type: 'codex-web-search' }
    }
    case 'codexResponseMessage': {
      const message = block as CodexPiResponseMessageContent
      return { ...message, type: 'codex-response-message' }
    }
    default: return undefined
  }
}

interface PendingReasoning {
  buffered: string
  visible: boolean
}

function closeReasoning(
  index: number,
  state: PendingReasoning | undefined,
  text: string,
): StreamChunk[] {
  if (text.trim() === '') {
    if (state?.visible !== true) return []
    throw new LlmError('Codex visible reasoning ended without content', 'PI_AI_ERROR')
  }
  return [
    ...state?.visible === true ? [] : [{
      type: 'block-start' as const,
      index,
      blockType: 'reasoning' as const,
    }],
    { type: 'block-end', index, block: { type: 'reasoning', text } },
  ]
}

// XXX(pi-ai upstream): pi-ai flattens the caught error to `error.message`
// (api/anthropic-messages.js: `errorMessage = error instanceof Error ?
// error.message : JSON.stringify(error)`), discarding the original Error and its
// `cause` chain before it reaches us. undici carries the actionable transport
// detail on `cause` (e.g. `SocketError: other side closed`) but hands the fetch
// wrapper a bare `terminated`, so we are left pattern-matching terse words here.
// If pi-ai ever forwards the original Error (or a fetch/dispatcher hook that lets
// us capture the cause ourselves), classify on `code`/`cause` instead of text.
function classifyPiAiError(message: string): string {
  if (/\b(?:401|403)\b/.test(message)) return 'AUTH'
  if (isQuotaExceededError(message)) return QUOTA_EXCEEDED_CODE
  if (/\b429\b|rate.?limit/i.test(message)) return 'RATE_LIMIT'
  if (/\b400\b|invalid.?request/i.test(message)) return 'INVALID_REQUEST'
  if (/\b5\d\d\b/.test(message)) return 'SERVER'
  if (/\btime(?:d)?\s*out\b|timeout/i.test(message)) return 'TIMEOUT'
  // A stream truncated before the provider's terminal event: each pi-ai provider
  // throws its own wording when the wire closes mid-response without a terminal
  // event (`… stream ended before message_stop`, `… before a terminal response
  // event`, `… ended without a terminal event`, `Stream ended without
  // finish_reason`). The connection dropped mid-response, so this is a transport
  // truncation, not a model-level error.
  if (/stream ended (?:before|without)\b/i.test(message)) return 'TRANSPORT'
  if (/\b(?:network|connection|socket|fetch)\b|\bECONN[A-Z]+\b/i.test(message)
    || /\b(?:other side closed|HTTP2 request did not get a response|WebSocket closed unexpectedly)\b/i.test(message)
    // undici renders a mid-stream socket drop as a bare `terminated` (its
    // `cause` — the real SocketError — was flattened away upstream); Node's
    // stream layer says `Premature close`.
    || /\bterminated\b|premature close/i.test(message)) {
    return 'TRANSPORT'
  }
  return 'PI_AI_ERROR'
}

/**
 * Map a terminal pi-ai event to the harness finish reason.
 * @param message - the assistant message carried by the `done` or `error` event.
 * @param contextWindow - resolved catalog capacity for usage-based overflow detection.
 * @returns the mapped harness reason. Recognized error text, `stop` usage above
 *   `contextWindow`, and zero-output `length` usage that fills the window map
 *   to `CONTEXT_WINDOW_EXCEEDED`; a `stop` with no content blocks maps to an
 *   `EMPTY_RESPONSE` error.
 */
export function mapStopReason(message: AssistantMessage, contextWindow?: number): FinishReason {
  const piAiOverflow = isContextOverflow(message, contextWindow)
  const harnessOverflow = message.stopReason === 'error'
    && message.errorMessage !== undefined
    && isContextWindowExceededError(message.errorMessage)
  if (piAiOverflow || harnessOverflow) {
    return {
      kind: 'error',
      failure: {
        message: message.errorMessage ?? `pi-ai detected context overflow for model "${message.model}"`,
        code: CONTEXT_WINDOW_EXCEEDED_CODE,
      },
    }
  }

  switch (message.stopReason) {
    case 'stop':
      // A terminal stop that produced no content blocks is a degenerate
      // provider completion, not a successful (empty) assistant message.
      if (message.content.length === 0) {
        return {
          kind: 'error',
          failure: {
            message: `model "${message.model}" returned a completed response with no content`,
            code: EMPTY_RESPONSE_CODE,
          },
        }
      }
      return { kind: 'stop' }
    case 'length': return { kind: 'max-tokens' }
    case 'toolUse': return { kind: 'tool-calls' }
    case 'aborted': return {
      kind: 'aborted',
      failure: { message: message.errorMessage ?? 'pi-ai stream aborted', code: 'ABORTED' },
    }
    case 'error': {
      const text = message.errorMessage ?? 'pi-ai stream error'
      return { kind: 'error', failure: { message: text, code: classifyPiAiError(text) } }
    }
  }
}

/**
 * Translate the pi-ai event stream into StreamChunks. pi-ai never throws
 * mid-stream — failures arrive as `error` events, which become error/aborted
 * `finish` chunks (the harness protocol's other error-delivery style).
 * @param events - one assistant turn's pi-ai event stream.
 * @param contextWindow - resolved catalog capacity for usage-based overflow detection.
 * @returns the harness chunks, ending with `usage` then `finish`; throws
 *   `LlmError` (`STREAM_CLOSED`) if the source ends without a terminal event.
 */
export async function* toStreamChunks(
  events: AsyncIterable<AssistantMessageEvent | CodexStructuredAssistantEvent>,
  contextWindow?: number,
): AsyncGenerator<StreamChunk> {
  // Preserve pi-ai contentIndex values for live block positions. DSH tolerates
  // gaps when an encrypted-only reasoning item is withheld from its renderer.
  const toolIds = new Map<number, { id: string; name: string }>()
  const openSearches = new Map<string, CodexWebSearchBlock>()
  const reasoning = new Map<number, PendingReasoning>()

  for await (const event of events) {
    switch (event.type) {
      case 'start':
        break
      case 'text_start':
        yield { type: 'block-start', index: event.contentIndex, blockType: 'text' }
        break
      case 'text_delta':
        yield { type: 'text-delta', index: event.contentIndex, text: event.delta }
        break
      case 'text_end':
        yield { type: 'block-end', index: event.contentIndex, block: { type: 'text', text: event.content } }
        break
      case 'thinking_start':
        reasoning.set(event.contentIndex, { buffered: '', visible: false })
        break
      case 'thinking_delta': {
        const state = reasoning.get(event.contentIndex) ?? { buffered: '', visible: false }
        state.buffered += event.delta
        reasoning.set(event.contentIndex, state)
        if (state.visible) {
          yield { type: 'reasoning-delta', index: event.contentIndex, text: event.delta }
          break
        }
        if (state.buffered.trim() === '') break
        state.visible = true
        yield { type: 'block-start', index: event.contentIndex, blockType: 'reasoning' }
        yield { type: 'reasoning-delta', index: event.contentIndex, text: state.buffered }
        break
      }
      case 'thinking_end': {
        const state = reasoning.get(event.contentIndex)
        reasoning.delete(event.contentIndex)
        for (const chunk of closeReasoning(event.contentIndex, state, event.content)) yield chunk
        break
      }
      case 'toolcall_start': {
        // The id/name live on the partial's content at this index.
        const partial = event.partial.content[event.contentIndex]
        const id = partial?.type === 'toolCall' ? partial.id : ''
        const name = partial?.type === 'toolCall' ? partial.name : ''
        toolIds.set(event.contentIndex, { id, name })
        yield { type: 'block-start', index: event.contentIndex, blockType: 'tool-call' }
        break
      }
      case 'toolcall_delta': {
        const known = toolIds.get(event.contentIndex)
        yield {
          type: 'tool-call-delta',
          index: event.contentIndex,
          id: CallId(known?.id ?? ''),
          ...known?.name !== undefined && known.name.length > 0 ? { name: known.name } : {},
          argumentsDelta: event.delta,
        }
        break
      }
      case 'toolcall_end':
        yield {
          type: 'block-end',
          index: event.contentIndex,
          block: {
            type: 'tool-call',
            id: CallId(event.toolCall.id),
            name: event.toolCall.name,
            // pi-ai hands back the PARSED arguments; the harness vocabulary
            // keeps the raw string.
            arguments: JSON.stringify(event.toolCall.arguments),
          },
        }
        break
      case 'codex_web_search_start': {
        const native = (event.partial.content as unknown[])[event.contentIndex] as {
          type: 'codexWebSearch'; id: string; status: CodexWebSearchBlock['status']; action?: CodexWebSearchBlock['action']
        } | undefined
        if (native?.type !== 'codexWebSearch') throw new LlmError('Codex search start has no content block', 'PI_AI_ERROR')
        const block: CodexWebSearchBlock = {
          type: 'codex-web-search', id: native.id, status: native.status,
          ...native.action === undefined ? {} : { action: native.action },
        }
        const frame = framedCodexBlock(block)
        openSearches.set(block.id, block)
        yield { type: 'block-start', index: event.contentIndex, blockType: 'text' }
        yield { type: 'block-end', index: event.contentIndex, block: frame }
        break
      }
      case 'codex_web_search_end': {
        const block: CodexWebSearchBlock = {
          type: 'codex-web-search', id: event.block.id, status: event.block.status,
          ...event.block.action === undefined ? {} : { action: event.block.action },
        }
        const open = openSearches.get(block.id)
        if (open === undefined) throw new LlmError('Codex search end has no open record', 'PI_AI_ERROR')
        openSearches.delete(block.id)
        const frame = framedCodexBlock(block)
        yield { type: 'block-start', index: event.contentIndex, blockType: 'text' }
        yield { type: 'block-end', index: event.contentIndex, block: frame }
        break
      }
      case 'codex_web_search_update': {
        const block: CodexWebSearchBlock = {
          type: 'codex-web-search', id: event.block.id, status: event.block.status,
          ...event.block.action === undefined ? {} : { action: event.block.action },
        }
        if (!openSearches.has(block.id)) throw new LlmError('Codex search update has no open record', 'PI_AI_ERROR')
        openSearches.set(block.id, block)
        const frame = framedCodexBlock(block)
        yield { type: 'block-start', index: event.contentIndex, blockType: 'text' }
        yield { type: 'block-end', index: event.contentIndex, block: frame }
        break
      }
      case 'codex_response_message': {
        const block = framedCodexBlock({
          type: 'codex-response-message', id: event.block.id, status: event.block.status,
          ...event.block.phase === undefined ? {} : { phase: event.block.phase },
          content: event.block.content,
        })
        yield { type: 'block-start', index: event.contentIndex, blockType: 'text' }
        yield { type: 'block-end', index: event.contentIndex, block }
        break
      }
      case 'done':
        {
          for (const [index, state] of reasoning) {
            const native = event.message.content[index]
            const text = native?.type === 'thinking'
              && (native.thinking.trim() !== '' || state.buffered.trim() === '')
              ? native.thinking
              : state.buffered
            for (const chunk of closeReasoning(index, state, text)) yield chunk
          }
          reasoning.clear()
          const structured = (event.message.content as unknown[])
            .map(structuredBlock)
            .filter((block): block is CodexStructuredBlock => block !== undefined)
          const sources = foreignSourcesBlock(structured)
          if (sources !== undefined) {
            const index = event.message.content.length
            yield { type: 'block-start', index, blockType: 'text' }
            yield { type: 'text-delta', index, text: sources.type === 'text' ? sources.text : '' }
            yield { type: 'block-end', index, block: sources }
          }
        yield { type: 'usage', usage: mapUsage(event.message.usage) }
        yield {
          type: 'finish',
          reason: mapStopReason(event.message, contextWindow),
          replayState: toCodexReplayState(event.message, sources !== undefined),
        }
        return
        }
      case 'error':
        {
          for (const [reasoningIndex, state] of reasoning) {
            const native = event.error.content[reasoningIndex]
            const text = native?.type === 'thinking'
              && (native.thinking.trim() !== '' || state.buffered.trim() === '')
              ? native.thinking
              : state.buffered
            for (const chunk of closeReasoning(reasoningIndex, state, text)) yield chunk
          }
          reasoning.clear()
          let index = event.error.content.length
          const failedSearches = event.error.stopReason === 'aborted'
            ? []
            : [...openSearches.values()].map(search => ({ ...search, status: 'failed' as const }))
          for (const search of failedSearches) {
            const frame = framedCodexBlock(search)
            yield { type: 'block-start', index, blockType: 'text' }
            yield { type: 'block-end', index, block: frame }
            index += 1
          }
          const replayMessage: AssistantMessage = failedSearches.length === 0
            ? event.error
            : {
              ...event.error,
              content: [
                ...event.error.content,
                ...failedSearches.map(search => ({ ...search, type: 'codexWebSearch' as const })),
              ],
            } as unknown as AssistantMessage
          yield { type: 'usage', usage: mapUsage(event.error.usage) }
          yield {
            type: 'finish',
            reason: mapStopReason(event.error, contextWindow),
            replayState: toCodexReplayState(replayMessage, false),
          }
        }
        return
      // no default: AssistantMessageEvent is pi-ai's closed union; a new
      // event type should fail compilation here via tsc's exhaustiveness
      // when one is added (switch covers all current variants).
    }
  }
  throw new LlmError('pi-ai event stream ended without done/error', 'STREAM_CLOSED')
}
