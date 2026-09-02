/** Durable replay for ordinary pi-ai blocks plus Codex hosted-search metadata. */

import { LlmError } from '@deepseek-ai/dsh-llm'
import type { Message, ModelMessageSource, ReplayEnvelope } from '@deepseek-ai/dsh-llm'
import type { Api, AssistantMessage, TextContent, Usage as PiUsage } from '@earendil-works/pi-ai'
import {
  encodeCodexReplayMarker,
  parseCodexResponseMessageBlock,
  parseCodexWebSearchBlock,
  structuredCodexBlock,
} from './structured-search.ts'
import type {
  CodexPiResponseMessageContent,
  CodexPiWebSearchContent,
  CodexResponseMessageBlock,
  CodexWebSearchBlock,
} from './structured-search.ts'

export interface CodexReplayReasoning {
  thinkingSignature?: string
  redacted?: boolean
}

interface CodexReplayPlacement {
  /** Empty native reasoning items immediately before this visible block. */
  leadingReasoning?: readonly CodexReplayReasoning[]
  /** Empty native reasoning items immediately after the final visible block. */
  trailingReasoning?: readonly CodexReplayReasoning[]
}

export type CodexReplayBlock = (
  | { type: 'text'; textSignature?: string }
  | { type: 'reasoning'; thinkingSignature?: string; redacted?: boolean }
  | { type: 'tool-call'; thoughtSignature?: string }
  | { type: 'codex-web-search'; block: CodexWebSearchBlock }
  | { type: 'codex-response-message'; block: CodexResponseMessageBlock }
  | { type: 'foreign-sources' }
) & CodexReplayPlacement

export interface CodexReplayResponse {
  kind: 'pi-ai' | 'dsh-codex-pi-ai'
  version: 1 | 2
  api: Api
  provider: string
  model: string
  responseModel?: string
  responseId?: string
  stopReason: AssistantMessage['stopReason']
  /** Empty reasoning when the native response has no visible blocks at all. */
  trailingReasoning?: readonly CodexReplayReasoning[]
}

interface CodexReplayState {
  response: CodexReplayResponse
  blocks: CodexReplayBlock[]
}

type ExtendedPiContent = AssistantMessage['content'][number]
  | CodexPiWebSearchContent
  | CodexPiResponseMessageContent

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

function parseArguments(raw: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(raw)
    if (record(parsed) !== undefined) return parsed as Record<string, unknown>
  } catch {}
  return {}
}

function emptyPiUsage(): PiUsage {
  return {
    input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  }
}

function webSearchBlock(block: CodexPiWebSearchContent): CodexWebSearchBlock {
  return { ...block, type: 'codex-web-search' }
}

function responseMessageBlock(block: CodexPiResponseMessageContent): CodexResponseMessageBlock {
  return { ...block, type: 'codex-response-message' }
}

function replayReasoning(block: Extract<AssistantMessage['content'][number], { type: 'thinking' }>): CodexReplayReasoning {
  return {
    ...block.thinkingSignature === undefined ? {} : { thinkingSignature: block.thinkingSignature },
    ...block.redacted === undefined ? {} : { redacted: block.redacted },
  }
}

function replayBlock(block: ExtendedPiContent): CodexReplayBlock {
  switch (block.type) {
    case 'text': return { type: 'text', ...block.textSignature === undefined ? {} : { textSignature: block.textSignature } }
    case 'thinking': return { type: 'reasoning', ...replayReasoning(block) }
    case 'toolCall': return { type: 'tool-call', ...block.thoughtSignature === undefined ? {} : { thoughtSignature: block.thoughtSignature } }
    case 'codexWebSearch': return { type: 'codex-web-search', block: webSearchBlock(block) }
    case 'codexResponseMessage': return { type: 'codex-response-message', block: responseMessageBlock(block) }
    default: throw new LlmError(`unsupported Codex pi-ai replay block ${(block as { type: string }).type}`, 'INVALID_REPLAY_STATE')
  }
}

/** Build metadata aligned with every DSH block emitted for this response. */
export function toCodexReplayState(message: AssistantMessage, appendForeignSources: boolean): ReplayEnvelope {
  let response: CodexReplayResponse = {
    kind: 'dsh-codex-pi-ai', version: 1, api: message.api,
    provider: message.provider, model: message.model,
    ...message.responseModel === undefined ? {} : { responseModel: message.responseModel },
    ...message.responseId === undefined ? {} : { responseId: message.responseId },
    stopReason: message.stopReason,
  }
  const blocks: CodexReplayBlock[] = []
  let leadingReasoning: CodexReplayReasoning[] = []
  for (const block of message.content as ExtendedPiContent[]) {
    if (block.type === 'thinking' && block.thinking.trim() === '') {
      leadingReasoning.push(replayReasoning(block))
      continue
    }
    blocks.push({
      ...replayBlock(block),
      ...leadingReasoning.length === 0 ? {} : { leadingReasoning },
    })
    leadingReasoning = []
  }
  if (leadingReasoning.length > 0) {
    const last = blocks.at(-1)
    if (last === undefined) response = { ...response, trailingReasoning: leadingReasoning }
    else blocks[blocks.length - 1] = { ...last, trailingReasoning: leadingReasoning }
  }
  if (appendForeignSources) blocks.push({ type: 'foreign-sources' })
  return { response, blocks }
}

/** Encode an in-memory structured pi-ai response for native payload conversion. */
export function withCodexPiReplayMarkers(message: AssistantMessage): AssistantMessage {
  const content = (message.content as ExtendedPiContent[]).map(block => {
    switch (block.type) {
      case 'codexWebSearch':
        return markerContent(encodeCodexReplayMarker({ kind: 'web-search', block: webSearchBlock(block) }))
      case 'codexResponseMessage':
        return markerContent(encodeCodexReplayMarker({ kind: 'response-message', block: responseMessageBlock(block) }))
      default:
        return block
    }
  }) as AssistantMessage['content']
  return { ...message, content }
}

function invalidReplay(message: string): never {
  throw new LlmError(`invalid Codex pi-ai replay state: ${message}`, 'INVALID_REPLAY_STATE')
}

function readReplayReasoning(value: unknown, field: string): readonly CodexReplayReasoning[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value)) return invalidReplay(`${field} must be an array`)
  return value.map((entry, position) => {
    const reasoning = record(entry)
    if (reasoning === undefined) return invalidReplay(`${field} item ${position} must be an object`)
    if (reasoning['thinkingSignature'] !== undefined && typeof reasoning['thinkingSignature'] !== 'string') {
      return invalidReplay(`${field} item ${position} thinkingSignature must be a string`)
    }
    if (reasoning['redacted'] !== undefined && typeof reasoning['redacted'] !== 'boolean') {
      return invalidReplay(`${field} item ${position} redacted must be boolean`)
    }
    return {
      ...reasoning['thinkingSignature'] === undefined ? {} : { thinkingSignature: reasoning['thinkingSignature'] },
      ...reasoning['redacted'] === undefined ? {} : { redacted: reasoning['redacted'] },
    } as CodexReplayReasoning
  })
}

function readReplayPlacement(block: Record<string, unknown>, index: number): CodexReplayPlacement {
  const leadingReasoning = readReplayReasoning(
    block['leadingReasoning'],
    `block ${index} leadingReasoning`,
  )
  const trailingReasoning = readReplayReasoning(
    block['trailingReasoning'],
    `block ${index} trailingReasoning`,
  )
  return {
    ...leadingReasoning === undefined ? {} : { leadingReasoning },
    ...trailingReasoning === undefined ? {} : { trailingReasoning },
  }
}

function readReplayBlock(value: unknown, index: number): CodexReplayBlock {
  const block = record(value)
  if (block === undefined || typeof block['type'] !== 'string') return invalidReplay(`block ${index} must be an object`)
  const placement = readReplayPlacement(block, index)
  switch (block['type']) {
    case 'text':
      if (block['textSignature'] !== undefined && typeof block['textSignature'] !== 'string') return invalidReplay(`block ${index} textSignature must be a string`)
      return {
        type: 'text',
        ...block['textSignature'] === undefined ? {} : { textSignature: block['textSignature'] },
        ...placement,
      }
    case 'reasoning':
      if (block['thinkingSignature'] !== undefined && typeof block['thinkingSignature'] !== 'string') return invalidReplay(`block ${index} thinkingSignature must be a string`)
      if (block['redacted'] !== undefined && typeof block['redacted'] !== 'boolean') return invalidReplay(`block ${index} redacted must be boolean`)
      return {
        type: 'reasoning',
        ...block['thinkingSignature'] === undefined ? {} : { thinkingSignature: block['thinkingSignature'] },
        ...block['redacted'] === undefined ? {} : { redacted: block['redacted'] },
        ...placement,
      }
    case 'tool-call':
      if (block['thoughtSignature'] !== undefined && typeof block['thoughtSignature'] !== 'string') return invalidReplay(`block ${index} thoughtSignature must be a string`)
      return {
        type: 'tool-call',
        ...block['thoughtSignature'] === undefined ? {} : { thoughtSignature: block['thoughtSignature'] },
        ...placement,
      }
    case 'codex-web-search': {
      const native = record(block['block'])
      return {
        type: 'codex-web-search',
        block: parseCodexWebSearchBlock({ ...(native ?? {}), type: 'web_search_call' }),
        ...placement,
      }
    }
    case 'codex-response-message': {
      const native = record(block['block'])
      const content = Array.isArray(native?.['content']) ? native.content.map(part => {
        const candidate = record(part)
        if (candidate?.['type'] !== 'output_text') return part
        const annotations = Array.isArray(candidate['annotations']) ? candidate.annotations.map(annotation => {
          const citation = record(annotation)
          return citation?.['type'] === 'url_citation' ? {
            type: 'url_citation', start_index: citation['startIndex'], end_index: citation['endIndex'],
            title: citation['title'], url: citation['url'],
          } : annotation
        }) : candidate['annotations']
        return { ...candidate, annotations }
      }) : native?.['content']
      return {
        type: 'codex-response-message',
        block: parseCodexResponseMessageBlock({ ...(native ?? {}), type: 'message', content }),
        ...placement,
      }
    }
    case 'foreign-sources': return { type: 'foreign-sources', ...placement }
    default: return invalidReplay(`block ${index} has an unknown type`)
  }
}

function readReplayState(value: unknown): CodexReplayState {
  const envelope = record(value)
  const response = record(envelope?.['response'])
  if (response === undefined) return invalidReplay('expected a response object')
  const kind = response['kind']
  const version = response['version']
  if (!((kind === 'pi-ai' && version === 2) || (kind === 'dsh-codex-pi-ai' && version === 1))) {
    return invalidReplay(`unsupported ${String(kind)} version ${String(version)}`)
  }
  for (const key of ['api', 'provider', 'model'] as const) {
    if (typeof response[key] !== 'string' || response[key].length === 0) return invalidReplay(`${key} must be a non-empty string`)
  }
  if (!['stop', 'length', 'toolUse', 'error', 'aborted'].includes(String(response['stopReason']))) return invalidReplay('unknown stopReason')
  if (response['responseModel'] !== undefined && typeof response['responseModel'] !== 'string') return invalidReplay('responseModel must be a string')
  if (response['responseId'] !== undefined && typeof response['responseId'] !== 'string') return invalidReplay('responseId must be a string')
  readReplayReasoning(response['trailingReasoning'], 'response trailingReasoning')
  const rawBlocks = envelope?.['blocks']
  if (!Array.isArray(rawBlocks)) return invalidReplay('blocks must be an array')
  return { response: response as unknown as CodexReplayResponse, blocks: rawBlocks.map(readReplayBlock) }
}

function foreignAssistant(message: Message): AssistantMessage {
  const source = message.source.kind === 'model' ? message.source : undefined
  const content: AssistantMessage['content'] = []
  for (const block of message.content) {
    if (block.type === 'text' && block.text === '' && block.codexStructured !== undefined) continue
    switch (block.type) {
      case 'text': content.push({ type: 'text', text: block.text }); break
      case 'reasoning': content.push({ type: 'thinking', thinking: block.text }); break
      case 'tool-call': content.push({ type: 'toolCall', id: block.id, name: block.name, arguments: parseArguments(block.arguments) }); break
      case 'image': throw new LlmError('pi-ai chat history cannot represent structured assistant image output', 'UNSUPPORTED_CONTENT')
      default: break
    }
  }
  return {
    role: 'assistant', content, api: 'dsh-foreign',
    provider: source?.provider ?? 'dsh-foreign', model: source?.model ?? 'dsh-foreign',
    usage: emptyPiUsage(), stopReason: content.some(piece => piece.type === 'toolCall') ? 'toolUse' : 'stop', timestamp: 0,
  }
}

function markerContent(text: string): TextContent { return { type: 'text', text } }

function sameStructuredBlock(left: CodexWebSearchBlock | CodexResponseMessageBlock, right: typeof left): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}

function emptyReasoningContent(reasoning: CodexReplayReasoning): Extract<AssistantMessage['content'][number], { type: 'thinking' }> {
  return {
    type: 'thinking',
    thinking: '',
    ...reasoning.thinkingSignature === undefined ? {} : { thinkingSignature: reasoning.thinkingSignature },
    ...reasoning.redacted === undefined ? {} : { redacted: reasoning.redacted },
  }
}

function replayedAssistant(message: Message, source: ModelMessageSource, rawState: unknown): AssistantMessage {
  const state = readReplayState(rawState)
  if (state.response.provider !== source.provider) return invalidReplay('provider does not match assistant source')
  if (state.response.model !== source.model) return invalidReplay('model does not match assistant source')
  if (state.blocks.length !== message.content.length) return invalidReplay('block count does not match assistant content')
  const content: AssistantMessage['content'] = []
  message.content.forEach((block, index) => {
    const replay = state.blocks[index]
    if (replay === undefined) return invalidReplay(`block ${index} metadata is absent`)
    content.push(...(replay.leadingReasoning ?? []).map(emptyReasoningContent))
    switch (replay.type) {
      case 'text':
        if (block.type !== 'text') return invalidReplay(`block ${index} does not match text content`)
        content.push({ type: 'text', text: block.text, ...replay.textSignature === undefined ? {} : { textSignature: replay.textSignature } })
        break
      case 'reasoning':
        if (block.type !== 'reasoning') return invalidReplay(`block ${index} does not match reasoning content`)
        content.push({ type: 'thinking', thinking: block.text, ...replay.thinkingSignature === undefined ? {} : { thinkingSignature: replay.thinkingSignature }, ...replay.redacted === undefined ? {} : { redacted: replay.redacted } })
        break
      case 'tool-call':
        if (block.type !== 'tool-call') return invalidReplay(`block ${index} does not match tool-call content`)
        content.push({ type: 'toolCall', id: block.id, name: block.name, arguments: parseArguments(block.arguments), ...replay.thoughtSignature === undefined ? {} : { thoughtSignature: replay.thoughtSignature } })
        break
      case 'codex-web-search':
        {
          const structured = structuredCodexBlock(block)
          if (structured?.type !== 'codex-web-search' || !sameStructuredBlock(structured, replay.block)) {
            return invalidReplay(`block ${index} does not match web-search content`)
          }
        }
        content.push(markerContent(encodeCodexReplayMarker({ kind: 'web-search', block: replay.block })))
        break
      case 'codex-response-message':
        {
          const structured = structuredCodexBlock(block)
          if (structured?.type !== 'codex-response-message' || !sameStructuredBlock(structured, replay.block)) {
            return invalidReplay(`block ${index} does not match response-message content`)
          }
        }
        content.push(markerContent(encodeCodexReplayMarker({ kind: 'response-message', block: replay.block })))
        break
      case 'foreign-sources':
        if (block.type !== 'text') return invalidReplay(`block ${index} does not match foreign-sources content`)
        content.push(markerContent(encodeCodexReplayMarker({ kind: 'foreign-sources' })))
        break
      default: replay satisfies never
    }
    content.push(...(replay.trailingReasoning ?? []).map(emptyReasoningContent))
  })
  content.push(...(state.response.trailingReasoning ?? []).map(emptyReasoningContent))
  return {
    role: 'assistant', content, api: state.response.api, provider: state.response.provider, model: state.response.model,
    ...state.response.responseModel === undefined ? {} : { responseModel: state.response.responseModel },
    ...state.response.responseId === undefined ? {} : { responseId: state.response.responseId },
    usage: emptyPiUsage(), stopReason: state.response.stopReason, timestamp: 0,
  }
}

/** Convert durable DSH history into exact Codex replay or safe foreign history. */
export function toCodexAssistant(message: Message, onDegrade?: (reason: string) => void): AssistantMessage {
  const source = message.source
  if (source.kind !== 'model' || source.replayState === undefined) return foreignAssistant(message)
  try {
    return replayedAssistant(message, source, source.replayState)
  } catch (error: unknown) {
    if (!(error instanceof LlmError) || error.code !== 'INVALID_REPLAY_STATE') throw error
    onDegrade?.(error.message)
    return foreignAssistant(message)
  }
}
