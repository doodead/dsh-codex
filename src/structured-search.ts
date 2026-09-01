/** Structured Codex hosted-search data preserved through stock DSH seams. */

import type { ContentBlock, TextBlock } from '@deepseek-ai/dsh-llm'

export const CODEX_WEB_SEARCH_STATUSES = [
  'in_progress',
  'searching',
  'completed',
  'failed',
] as const

export type CodexWebSearchStatus = typeof CODEX_WEB_SEARCH_STATUSES[number]

export interface CodexWebSearchSource {
  readonly type: 'url'
  readonly url: string
}

export type CodexWebSearchAction =
  | {
    readonly type: 'search'
    readonly query: string
    readonly queries?: readonly string[]
    readonly sources?: readonly CodexWebSearchSource[]
  }
  | { readonly type: 'open_page'; readonly url?: string | null }
  | { readonly type: 'find_in_page'; readonly pattern: string; readonly url: string }

/** Atomic durable representation of one provider-hosted web-search call. */
export interface CodexWebSearchBlock {
  readonly type: 'codex-web-search'
  readonly id: string
  readonly status: CodexWebSearchStatus
  readonly action: CodexWebSearchAction
}

export interface CodexUrlCitation {
  readonly type: 'url_citation'
  readonly startIndex: number
  readonly endIndex: number
  readonly title: string
  readonly url: string
}

export type CodexResponseContentPart =
  | {
    readonly type: 'output_text'
    readonly text: string
    readonly annotations: readonly CodexUrlCitation[]
  }
  | { readonly type: 'refusal'; readonly refusal: string }

/** Provider-native message metadata kept beside ordinary visible DSH text. */
export interface CodexResponseMessageBlock {
  readonly type: 'codex-response-message'
  readonly id: string
  readonly status: 'in_progress' | 'completed' | 'incomplete'
  readonly phase?: 'commentary' | 'final_answer'
  readonly content: readonly CodexResponseContentPart[]
}

export type CodexStructuredBlock = CodexWebSearchBlock | CodexResponseMessageBlock

export type CodexStructuredFrame =
  | { readonly version: 1; readonly kind: 'web-search'; readonly block: CodexWebSearchBlock }
  | { readonly version: 1; readonly kind: 'response-message'; readonly block: CodexResponseMessageBlock }

declare module '@deepseek-ai/dsh-llm' {
  interface TextBlock {
    /** Invisible, schema-validated provider metadata carried beside empty text. */
    readonly codexStructured?: CodexStructuredFrame
  }
}

export interface CodexPiWebSearchContent extends Omit<CodexWebSearchBlock, 'type'> {
  readonly type: 'codexWebSearch'
}

export interface CodexPiResponseMessageContent extends Omit<CodexResponseMessageBlock, 'type'> {
  readonly type: 'codexResponseMessage'
}

export type CodexPiContent = CodexPiWebSearchContent | CodexPiResponseMessageContent

export const CODEX_REPLAY_MARKER_OPEN = '<dsh-codex-replay-5d54ff34-v1>'
export const CODEX_REPLAY_MARKER_CLOSE = '</dsh-codex-replay-5d54ff34-v1>'

export type CodexReplayMarker =
  | { readonly kind: 'web-search'; readonly block: CodexWebSearchBlock }
  | { readonly kind: 'response-message'; readonly block: CodexResponseMessageBlock }
  | { readonly kind: 'foreign-sources' }

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

function stringArray(value: unknown, field: string): readonly string[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value) || value.some(entry => typeof entry !== 'string')) {
    throw new TypeError(`Codex web search ${field} is malformed`)
  }
  return value
}

/** Parse one web_search_call item exhaustively and fail closed on new wire variants. */
export function parseCodexWebSearchBlock(value: unknown): CodexWebSearchBlock {
  const item = record(value)
  if (item?.['type'] !== 'web_search_call' || typeof item['id'] !== 'string') {
    throw new TypeError('Codex web search item is malformed')
  }
  const status = item['status']
  if (!(CODEX_WEB_SEARCH_STATUSES as readonly unknown[]).includes(status)) {
    throw new TypeError(`unsupported Codex web search status ${JSON.stringify(status)}`)
  }
  const rawAction = record(item['action'])
  if (rawAction === undefined) throw new TypeError('Codex web search action is malformed')
  const actionType = rawAction?.['type']
  let action: CodexWebSearchAction
  switch (actionType) {
    case 'search': {
      if (typeof rawAction['query'] !== 'string') throw new TypeError('Codex search query is malformed')
      const queries = stringArray(rawAction['queries'], 'queries')
      const rawSources = rawAction['sources']
      let sources: CodexWebSearchSource[] | undefined
      if (rawSources !== undefined) {
        if (!Array.isArray(rawSources)) throw new TypeError('Codex web search sources are malformed')
        sources = rawSources.map(source => {
          const candidate = record(source)
          if (candidate?.['type'] !== 'url' || typeof candidate['url'] !== 'string') {
            throw new TypeError('Codex web search source is malformed')
          }
          return { type: 'url', url: candidate['url'] }
        })
      }
      action = {
        type: 'search',
        query: rawAction['query'],
        ...queries === undefined ? {} : { queries },
        ...sources === undefined ? {} : { sources },
      }
      break
    }
    case 'open_page': {
      const url = rawAction['url']
      if (url !== undefined && url !== null && typeof url !== 'string') {
        throw new TypeError('Codex open-page URL is malformed')
      }
      action = { type: 'open_page', ...url === undefined ? {} : { url } }
      break
    }
    case 'find_in_page': {
      if (typeof rawAction['pattern'] !== 'string' || typeof rawAction['url'] !== 'string') {
        throw new TypeError('Codex find-in-page action is malformed')
      }
      action = { type: 'find_in_page', pattern: rawAction['pattern'], url: rawAction['url'] }
      break
    }
    default:
      throw new TypeError(`unsupported Codex web search action ${JSON.stringify(actionType)}`)
  }
  return { type: 'codex-web-search', id: item['id'], status: status as CodexWebSearchStatus, action }
}

function parseUrlCitation(value: unknown): CodexUrlCitation | undefined {
  const annotation = record(value)
  if (annotation?.['type'] !== 'url_citation') return undefined
  if (!Number.isSafeInteger(annotation['start_index'])
    || !Number.isSafeInteger(annotation['end_index'])
    || (annotation['start_index'] as number) < 0
    || (annotation['end_index'] as number) < (annotation['start_index'] as number)
    || typeof annotation['title'] !== 'string'
    || typeof annotation['url'] !== 'string') {
    throw new TypeError('Codex URL citation is malformed')
  }
  return {
    type: 'url_citation',
    startIndex: annotation['start_index'] as number,
    endIndex: annotation['end_index'] as number,
    title: annotation['title'],
    url: annotation['url'],
  }
}

/** Parse the native message item that owns output_text URL annotations. */
export function parseCodexResponseMessageBlock(value: unknown): CodexResponseMessageBlock {
  const item = record(value)
  if (item?.['type'] !== 'message' || typeof item['id'] !== 'string' || !Array.isArray(item['content'])) {
    throw new TypeError('Codex response message item is malformed')
  }
  const status = item['status']
  if (status !== 'in_progress' && status !== 'completed' && status !== 'incomplete') {
    throw new TypeError(`unsupported Codex response message status ${JSON.stringify(status)}`)
  }
  const phase = item['phase']
  if (phase !== undefined && phase !== null && phase !== 'commentary' && phase !== 'final_answer') {
    throw new TypeError(`unsupported Codex response message phase ${JSON.stringify(phase)}`)
  }
  const content: CodexResponseContentPart[] = item['content'].map(rawPart => {
    const part = record(rawPart)
    if (part?.['type'] === 'output_text') {
      if (typeof part['text'] !== 'string' || !Array.isArray(part['annotations'])) {
        throw new TypeError('Codex output_text part is malformed')
      }
      const text = part['text']
      const annotations = part['annotations'].map(parseUrlCitation).filter(
        (citation): citation is CodexUrlCitation => citation !== undefined,
      )
      if (annotations.some(citation => citation.endIndex > text.length)) {
        throw new TypeError('Codex URL citation offsets exceed output text')
      }
      return {
        type: 'output_text',
        text,
        annotations,
      }
    }
    if (part?.['type'] === 'refusal' && typeof part['refusal'] === 'string') {
      return { type: 'refusal', refusal: part['refusal'] }
    }
    throw new TypeError(`unsupported Codex response message content ${JSON.stringify(part?.['type'])}`)
  })
  return {
    type: 'codex-response-message',
    id: item['id'],
    status,
    ...phase === undefined || phase === null ? {} : { phase },
    content,
  }
}

/** Validate the durable metadata frame before replay or rendering. */
export function parseCodexStructuredFrame(value: unknown): CodexStructuredFrame {
  const frame = record(value)
  if (frame?.['version'] !== 1) throw new TypeError('unsupported Codex structured frame version')
  const block = record(frame['block'])
  switch (frame['kind']) {
    case 'web-search':
      return {
        version: 1,
        kind: 'web-search',
        block: parseCodexWebSearchBlock({ ...(block ?? {}), type: 'web_search_call' }),
      }
    case 'response-message': {
      const content = Array.isArray(block?.['content']) ? block.content.map(part => {
        const candidate = record(part)
        if (candidate?.['type'] !== 'output_text') return part
        return {
          ...candidate,
          annotations: Array.isArray(candidate['annotations'])
            ? candidate.annotations.map(annotation => {
              const citation = record(annotation)
              return citation?.['type'] === 'url_citation'
                ? {
                  type: 'url_citation',
                  start_index: citation['startIndex'],
                  end_index: citation['endIndex'],
                  title: citation['title'],
                  url: citation['url'],
                }
                : annotation
            })
            : candidate['annotations'],
        }
      }) : block?.['content']
      return {
        version: 1,
        kind: 'response-message',
        block: parseCodexResponseMessageBlock({ ...(block ?? {}), type: 'message', content }),
      }
    }
    default: throw new TypeError('unsupported Codex structured frame kind')
  }
}

/** Carry one structured record without producing visible assistant prose. */
export function framedCodexBlock(block: CodexStructuredBlock): TextBlock {
  const frame: CodexStructuredFrame = block.type === 'codex-web-search'
    ? { version: 1, kind: 'web-search', block }
    : { version: 1, kind: 'response-message', block }
  return { type: 'text', text: '', codexStructured: frame }
}

/** Read and validate structured metadata from one otherwise-empty text block. */
export function structuredCodexBlock(block: ContentBlock): CodexStructuredBlock | undefined {
  if (block.type !== 'text' || block.codexStructured === undefined) return undefined
  if (block.text !== '') throw new TypeError('Codex structured framing must not contain visible text')
  return parseCodexStructuredFrame(block.codexStructured).block
}

export function encodeCodexReplayMarker(marker: CodexReplayMarker): string {
  return `${CODEX_REPLAY_MARKER_OPEN}${JSON.stringify(marker)}${CODEX_REPLAY_MARKER_CLOSE}`
}

export function decodeCodexReplayMarker(text: string): CodexReplayMarker | undefined {
  if (!text.startsWith(CODEX_REPLAY_MARKER_OPEN) || !text.endsWith(CODEX_REPLAY_MARKER_CLOSE)) return undefined
  const raw = text.slice(CODEX_REPLAY_MARKER_OPEN.length, -CODEX_REPLAY_MARKER_CLOSE.length)
  const parsed = record(JSON.parse(raw))
  switch (parsed?.['kind']) {
    case 'web-search':
      return { kind: 'web-search', block: parseCodexWebSearchBlock({
        ...(record(parsed['block']) ?? {}), type: 'web_search_call',
      }) }
    case 'response-message': {
      const block = record(parsed['block'])
      return { kind: 'response-message', block: parseCodexResponseMessageBlock({
        ...(block ?? {}), type: 'message',
        content: Array.isArray(block?.['content'])
          ? block.content.map(part => {
            const candidate = record(part)
            return candidate?.['type'] === 'output_text'
              ? {
                ...candidate,
                annotations: Array.isArray(candidate['annotations'])
                  ? candidate.annotations.map(annotation => {
                    const citation = record(annotation)
                    return citation?.['type'] === 'url_citation'
                      ? {
                        type: 'url_citation',
                        start_index: citation['startIndex'],
                        end_index: citation['endIndex'],
                        title: citation['title'],
                        url: citation['url'],
                      }
                      : annotation
                  })
                  : candidate['annotations'],
              }
              : part
          })
          : block?.['content'],
      }) }
    }
    case 'foreign-sources': return { kind: 'foreign-sources' }
    default: throw new TypeError('Codex replay marker is malformed')
  }
}

function markerFromResponseItem(item: unknown): CodexReplayMarker | undefined {
  const candidate = record(item)
  if (candidate?.['type'] !== 'message' || candidate['role'] !== 'assistant'
    || !Array.isArray(candidate['content']) || candidate['content'].length !== 1) return undefined
  const content = record(candidate['content'][0])
  return content?.['type'] === 'output_text' && typeof content['text'] === 'string'
    ? decodeCodexReplayMarker(content['text'])
    : undefined
}

function nativeWebSearchItem(block: CodexWebSearchBlock): Record<string, unknown> {
  return {
    type: 'web_search_call',
    id: block.id,
    status: block.status,
    action: block.action,
  }
}

function nativeMessageItem(block: CodexResponseMessageBlock): Record<string, unknown> {
  return {
    type: 'message',
    id: block.id,
    role: 'assistant',
    status: block.status,
    ...block.phase === undefined ? {} : { phase: block.phase },
    content: block.content.map(part => part.type === 'refusal'
      ? part
      : {
        type: 'output_text',
        text: part.text,
        annotations: part.annotations.map(citation => ({
          type: 'url_citation',
          start_index: citation.startIndex,
          end_index: citation.endIndex,
          title: citation.title,
          url: citation.url,
        })),
      }),
  }
}

/** Replace adapter-only marker messages with exact native Responses items. */
export function expandCodexReplayMarkers(input: readonly unknown[]): unknown[] {
  const expanded: unknown[] = []
  const searchItems = new Map<string, number>()
  for (const item of input) {
    const marker = markerFromResponseItem(item)
    if (marker === undefined) {
      expanded.push(item)
      continue
    }
    switch (marker.kind) {
      case 'web-search':
        {
          const previous = searchItems.get(marker.block.id)
          if (previous === undefined) {
            if (marker.block.status === 'in_progress' || marker.block.status === 'searching') {
              searchItems.set(marker.block.id, expanded.length)
            }
            expanded.push(nativeWebSearchItem(marker.block))
          } else {
            expanded[previous] = nativeWebSearchItem(marker.block)
            if (marker.block.status === 'completed' || marker.block.status === 'failed') {
              searchItems.delete(marker.block.id)
            }
          }
        }
        break
      case 'response-message': {
        const previous = record(expanded.at(-1))
        if (previous?.['type'] !== 'message' || previous['role'] !== 'assistant') {
          throw new TypeError('Codex response-message replay marker has no preceding assistant message')
        }
        const previousContent = Array.isArray(previous['content']) ? previous['content'] : []
        const previousText = previousContent.map(part => {
          const content = record(part)
          return content?.['type'] === 'output_text' && typeof content['text'] === 'string'
            ? content['text']
            : content?.['type'] === 'refusal' && typeof content['refusal'] === 'string'
              ? content['refusal']
              : ''
        }).join('')
        const nativeText = marker.block.content.map(part => part.type === 'output_text' ? part.text : part.refusal).join('')
        if (previousText !== nativeText) {
          throw new TypeError('Codex response-message replay text does not match durable content')
        }
        expanded[expanded.length - 1] = nativeMessageItem(marker.block)
        break
      }
      case 'foreign-sources':
        break
      default:
        marker satisfies never
    }
  }
  return expanded
}

export function isSafeHttpUrl(value: string): boolean {
  try {
    const protocol = new URL(value).protocol
    return protocol === 'http:' || protocol === 'https:'
  } catch {
    return false
  }
}

function safeMarkdownUrl(value: string): string | undefined {
  try {
    const parsed = new URL(value)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return undefined
    return parsed.href.replaceAll('<', '%3C').replaceAll('>', '%3E')
  } catch {
    return undefined
  }
}

function escapeMarkdownLabel(value: string): string {
  return value.replace(/([\\\[\]])/g, '\\$1').replace(/[\r\n]+/g, ' ').trim()
}

/** Build provider-neutral citation fallback prose in first-citation order. */
export function foreignSourcesBlock(blocks: readonly CodexStructuredBlock[]): TextBlock | undefined {
  const seen = new Set<string>()
  const sources: Array<{ title: string; url: string }> = []
  for (const block of blocks) {
    if (block.type !== 'codex-response-message') continue
    for (const part of block.content) {
      if (part.type !== 'output_text') continue
      for (const citation of part.annotations) {
        const safeUrl = safeMarkdownUrl(citation.url)
        if (safeUrl === undefined || seen.has(safeUrl)) continue
        seen.add(safeUrl)
        sources.push({ title: citation.title || citation.url, url: safeUrl })
      }
    }
  }
  if (sources.length === 0) return undefined
  return {
    type: 'text',
    text: `\n\n### Sources\n\n${sources.map(source => `- [${escapeMarkdownLabel(source.title)}](<${source.url}>)`).join('\n')}`,
  }
}
