import type {
  ChatConversationViewNode,
  ClientContext,
  ConversationLocation,
  ConversationNodeDefinition,
} from '@deepseek-ai/dsh-client-runtime/client'
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import type { PropsLocale, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { ReactNode } from 'react'
import type {
  CodexResponseMessageBlock,
  CodexUrlCitation,
  CodexWebSearchAction,
  CodexWebSearchBlock,
} from '../structured-search.ts'
import { isSafeHttpUrl, structuredCodexBlock } from '../structured-search.ts'

export interface CodexWebSearchCallView {
  readonly key: string
  readonly status: CodexWebSearchBlock['status']
  readonly action?: CodexWebSearchAction
}

export interface CodexWebSearchChatData {
  readonly status: 'running' | 'settled' | 'interrupted'
  readonly calls: readonly CodexWebSearchCallView[]
  readonly citations: readonly CodexUrlCitation[]
}

declare module '@deepseek-ai/dsh-client-ui-conversation/client' {
  interface ChatNodeDataMap {
    /** One companion hosted-search row for an assistant step. */
    'codex-web-search': CodexWebSearchChatData
  }
}

interface SearchState {
  readonly calls: ReadonlyMap<string, CodexWebSearchCallView>
  readonly messages: ReadonlyMap<number, CodexResponseMessageBlock>
  readonly firstSeq?: number
  readonly finalized: boolean
}

function initialState(): SearchState {
  return { calls: new Map(), messages: new Map(), finalized: false }
}

function structuredBlocks(content: readonly ContentBlock[]): {
  calls: ReadonlyMap<string, CodexWebSearchCallView>
  messages: ReadonlyMap<number, CodexResponseMessageBlock>
} {
  const calls = new Map<string, CodexWebSearchCallView>()
  const messages = new Map<number, CodexResponseMessageBlock>()
  content.forEach((block, index) => {
    let structured: ReturnType<typeof structuredCodexBlock>
    try { structured = structuredCodexBlock(block) } catch { return }
    if (structured?.type === 'codex-web-search') {
      calls.set(structured.id, {
        key: structured.id,
        status: structured.status,
        ...structured.action === undefined ? {} : { action: structured.action },
      })
    } else if (structured?.type === 'codex-response-message') {
      messages.set(index, structured)
    }
  })
  return { calls, messages }
}

function updateChunk(state: SearchState, event: Extract<Parameters<ConversationNodeDefinition<SearchState>['match']>[0], { type: 'assistant/chunk' }>): SearchState {
  const chunk = event.data.chunk
  if (chunk.type === 'finish' && chunk.reason.kind === 'error') {
    return {
      ...state,
      calls: new Map([...state.calls].map(([id, call]) => [
        id,
        call.status === 'in_progress' || call.status === 'searching'
          ? { ...call, status: 'failed' }
          : call,
      ])),
      finalized: true,
    }
  }
  if (chunk.type === 'finish' && chunk.reason.kind === 'aborted') {
    return { ...state, finalized: false }
  }
  if (chunk.type !== 'block-end') return state
  let structured: ReturnType<typeof structuredCodexBlock>
  try { structured = structuredCodexBlock(chunk.block) } catch { return state }
  if (structured?.type === 'codex-web-search') {
    const calls = new Map(state.calls)
    calls.set(structured.id, {
      key: structured.id,
      status: structured.status,
      ...structured.action === undefined ? {} : { action: structured.action },
    })
    return {
      ...state, calls,
      firstSeq: state.firstSeq ?? event.seq,
    }
  }
  if (structured?.type === 'codex-response-message') {
    const messages = new Map(state.messages)
    messages.set(chunk.index, structured)
    return { ...state, messages }
  }
  return state
}

function locationClosed(location: ConversationLocation): boolean {
  if (location.kind === 'step') return location.step.status === 'closed' || location.turn.status === 'closed'
  return location.kind === 'turn' && location.turn.status === 'closed'
}

function citationsOf(messages: ReadonlyMap<number, CodexResponseMessageBlock>): CodexUrlCitation[] {
  const seen = new Set<string>()
  const result: CodexUrlCitation[] = []
  for (const message of messages.values()) {
    for (const part of message.content) {
      if (part.type !== 'output_text') continue
      for (const citation of part.annotations) {
        const key = `${citation.url}\u0000${citation.startIndex}\u0000${citation.endIndex}`
        if (seen.has(key)) continue
        seen.add(key)
        result.push(citation)
      }
    }
  }
  return result
}

export const codexWebSearchDefinition: ConversationNodeDefinition<SearchState> = {
  kind: 'codex-web-search',
  target: 'chat',
  match: event => {
    if (event.type === 'step/start') return { id: `${event.data.turn}:${event.data.step}`, role: 'start' }
    if (event.type === 'assistant/chunk' || event.type === 'assistant/message') {
      return { id: `${event.data.turn}:${event.data.step}`, role: 'update' }
    }
    return null
  },
  start: () => initialState(),
  update: (context, match) => {
    if (match.event.type === 'assistant/chunk') return updateChunk(context.state, match.event)
    if (match.event.type === 'assistant/message') {
      const blocks = structuredBlocks(match.event.data.message.content)
      return {
        ...context.state,
        calls: blocks.calls,
        messages: blocks.messages,
        finalized: [...blocks.calls.values()].every(
          call => call.status === 'completed' || call.status === 'failed',
        ),
      }
    }
    return context.state
  },
  publication: match => match.event.type === 'assistant/chunk' ? 'animation-frame' : 'immediate',
  buildViewNode: (context): ChatConversationViewNode | null => {
    if (context.state === undefined || context.state.calls.size === 0) return null
    const location = context.start?.location ?? context.matches.at(-1)?.location ?? { kind: 'unresolved' }
    const interrupted = !context.state.finalized && locationClosed(location)
    return {
      key: context.key,
      kind: 'codex-web-search',
      id: context.id,
      target: 'chat',
      anchorSeq: context.state.firstSeq ?? context.start?.event.seq ?? 0,
      location,
      visibility: 'visible',
      data: {
        status: interrupted ? 'interrupted' : context.state.finalized ? 'settled' : 'running',
        calls: [...context.state.calls.values()],
        citations: citationsOf(context.state.messages),
      },
    }
  },
}

type SearchNodeProps =
  PropsRuntime<'conversation.chat.node', 'codex-web-search'>
  & PropsLocale<'settings.openai-codex'>

function actionDetails(action: CodexWebSearchAction | undefined, t: SearchNodeProps['t']): ReactNode {
  if (action === undefined) return t('webSearchSearching')
  switch (action.type) {
    case 'search': {
      const queries = [...new Set([action.query, ...(action.queries ?? [])])]
      return queries.length === 1
        ? action.query
        : <><span>{action.query}</span><ul>{queries.map(query => <li key={query}>{query}</li>)}</ul></>
    }
    case 'open_page': return action.url === undefined || action.url === null
      ? t('webSearchOpenPage')
      : link(action.url, action.url)
    case 'find_in_page': return <>
      <span>{t('webSearchFindInPage', { pattern: action.pattern })}</span>{' '}
      {link(action.url, action.url)}
    </>
    default: return action satisfies never
  }
}

function link(url: string, label: string) {
  return isSafeHttpUrl(url)
    ? <a href={url} target="_blank" rel="noreferrer">{label}</a>
    : <span>{label}</span>
}

/** Compact companion row; native assistant Markdown remains owned by DSH. */
export function CodexWebSearchNode({ node, t }: SearchNodeProps) {
  const failed = node.data.calls.some(call => call.status === 'failed')
  const title = node.data.status === 'running'
    ? t('webSearchSearching')
    : node.data.status === 'interrupted'
      ? t('webSearchInterrupted')
      : failed
        ? t('webSearchFailed', { count: node.data.calls.length })
        : t('webSearchComplete', { count: node.data.calls.length })
  return (
    <details data-codex-web-search data-status={node.data.status} open={node.data.status === 'running'}>
      <summary>{title}</summary>
      <ol>
        {node.data.calls.map(call => (
          <li key={call.key} data-search-status={call.status}>
            {actionDetails(call.action, t)}
            {call.action?.type === 'search' && call.action.sources !== undefined && (
              <ul>{call.action.sources.map((source, index) => (
                <li key={`${source.url}-${index}`}>{link(source.url, source.url)}</li>
              ))}</ul>
            )}
          </li>
        ))}
      </ol>
      {node.data.citations.length > 0 && (
        <div>
          <strong>{t('webSearchCitations')}</strong>
          <ul>{node.data.citations.map((citation, index) => (
            <li key={`${citation.url}-${citation.startIndex}-${index}`}>
              {link(citation.url, citation.title || citation.url)}
            </li>
          ))}</ul>
        </div>
      )}
    </details>
  )
}

/** Register the Definition and keyed renderer through public client seams. */
export function registerCodexWebSearchNode(ctx: ClientContext, namespace: 'settings.openai-codex'): void {
  ctx.conversationEvents.register(codexWebSearchDefinition)
  ctx.slots.inject('conversation.chat.node', () => ctx.slots.register({
    name: 'conversation.chat.node', key: 'codex-web-search', locale: namespace,
  }, CodexWebSearchNode))
}
