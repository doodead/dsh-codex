/**
 * Structured-event observer layered around pi-ai 0.82.1's Responses parser.
 *
 * The upstream parser remains authoritative for text, reasoning, tools, usage,
 * and terminal behavior. This observer retains only the hosted-search records
 * that upstream otherwise discards.
 */

import { processResponsesStream as processUpstreamResponsesStream } from '@earendil-works/pi-ai/api/openai-responses-shared'
import type {
  Api,
  AssistantMessage,
  AssistantMessageEventStream,
  Model,
} from '@earendil-works/pi-ai'
import type { OpenAIResponsesStreamOptions } from '@earendil-works/pi-ai/api/openai-responses-shared'
import {
  parseCodexResponseMessageBlock,
  parseCodexWebSearchBlock,
} from '../../structured-search.ts'
import type {
  CodexPiResponseMessageContent,
  CodexPiWebSearchContent,
} from '../../structured-search.ts'

type JsonRecord = Record<string, unknown>

export type CodexStructuredAssistantEvent =
  | {
    readonly type: 'codex_web_search_start'
    readonly contentIndex: number
    readonly partial: AssistantMessage
  }
  | {
    readonly type: 'codex_web_search_end'
    readonly contentIndex: number
    readonly block: CodexPiWebSearchContent
    readonly partial: AssistantMessage
  }
  | {
    readonly type: 'codex_web_search_update'
    readonly contentIndex: number
    readonly block: CodexPiWebSearchContent
    readonly partial: AssistantMessage
  }
  | {
    readonly type: 'codex_response_message'
    readonly contentIndex: number
    readonly block: CodexPiResponseMessageContent
    readonly partial: AssistantMessage
  }

function record(value: unknown): JsonRecord | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as JsonRecord
    : undefined
}

function pushStructured(
  stream: AssistantMessageEventStream,
  event: CodexStructuredAssistantEvent,
): void {
  ;(stream as unknown as { push(event: CodexStructuredAssistantEvent): void }).push(event)
}

/** Preserve hosted-search output while delegating ordinary parsing upstream. */
export async function processStructuredResponsesStream<TApi extends Api>(
  source: AsyncIterable<unknown>,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  model: Model<TApi>,
  options?: OpenAIResponsesStreamOptions,
): Promise<void> {
  const searchSlots = new Map<number, { id: string; block: CodexPiWebSearchContent }>()

  const observed = async function* (): AsyncGenerator<unknown> {
    for await (const event of source) {
      const raw = record(event)
      if (raw?.['type'] === 'response.output_item.added') {
        const item = record(raw['item'])
        if (item?.['type'] === 'web_search_call') {
          if (!Number.isSafeInteger(raw['output_index'])) {
            throw new TypeError('Codex web search output index is malformed')
          }
          const parsed = parseCodexWebSearchBlock(item)
          const block: CodexPiWebSearchContent = { ...parsed, type: 'codexWebSearch' }
          const contentIndex = output.content.length
          ;(output.content as unknown[]).push(block)
          searchSlots.set(raw['output_index'] as number, { id: block.id, block })
          pushStructured(stream, { type: 'codex_web_search_start', contentIndex, partial: output })
        }
      }

      if (raw?.['type'] === 'response.web_search_call.searching') {
        if (!Number.isSafeInteger(raw['output_index']) || typeof raw['item_id'] !== 'string') {
          throw new TypeError('Codex web search status event is malformed')
        }
        const slot = searchSlots.get(raw['output_index'] as number)
        if (slot === undefined || slot.id !== raw['item_id']) {
          throw new TypeError('Codex web search status has no matching call')
        }
        const block: CodexPiWebSearchContent = { ...slot.block, status: 'searching' }
        const contentIndex = output.content.length
        ;(output.content as unknown[]).push(block)
        slot.block = block
        pushStructured(stream, {
          type: 'codex_web_search_update', contentIndex, block, partial: output,
        })
      }

      yield event

      if (raw?.['type'] !== 'response.output_item.done') continue
      const item = record(raw['item'])
      if (item?.['type'] === 'web_search_call') {
        if (!Number.isSafeInteger(raw['output_index'])) {
          throw new TypeError('Codex web search output index is malformed')
        }
        const outputIndex = raw['output_index'] as number
        const slot = searchSlots.get(outputIndex)
        if (slot === undefined) throw new TypeError('Codex web search completed without a matching start')
        const parsed = parseCodexWebSearchBlock(item)
        if (parsed.id !== slot.id) throw new TypeError('Codex web search completion id does not match its start')
        const block: CodexPiWebSearchContent = { ...parsed, type: 'codexWebSearch' }
        const contentIndex = output.content.length
        ;(output.content as unknown[]).push(block)
        pushStructured(stream, {
          type: 'codex_web_search_end',
          contentIndex,
          block,
          partial: output,
        })
        searchSlots.delete(outputIndex)
        continue
      }
      if (item?.['type'] !== 'message') continue
      const parsed = parseCodexResponseMessageBlock(item)
      const block: CodexPiResponseMessageContent = { ...parsed, type: 'codexResponseMessage' }
      const contentIndex = output.content.length
      ;(output.content as unknown[]).push(block)
      pushStructured(stream, {
        type: 'codex_response_message',
        contentIndex,
        block,
        partial: output,
      })
    }
  }

  await processUpstreamResponsesStream(
    observed() as never,
    output,
    stream,
    model,
    options,
  )
}
