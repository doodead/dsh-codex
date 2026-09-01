/** Request-local translation from Harness web_search to Codex hosted search. */

import type { Tool } from '@earendil-works/pi-ai'

export const NATIVE_WEB_SEARCH_MODES = ['cached', 'indexed', 'live'] as const
export type NativeWebSearchMode = typeof NATIVE_WEB_SEARCH_MODES[number]

export const NATIVE_WEB_SEARCH_CONTEXT_SIZES = ['omit', 'low', 'medium', 'high'] as const
export type NativeWebSearchContextSize = typeof NATIVE_WEB_SEARCH_CONTEXT_SIZES[number]

type JsonRecord = Record<string, unknown>

export interface NativeWebSearchTransformOptions {
  readonly harnessTools: readonly Pick<Tool, 'name'>[] | undefined
  readonly enabled: boolean
  readonly mode: NativeWebSearchMode
  readonly contextSize: NativeWebSearchContextSize
  readonly alwaysAvailable: boolean
}

export interface HostedNativeWebSearchTool {
  readonly type: 'web_search'
  readonly external_web_access: boolean
  readonly indexed_web_access?: true
  readonly search_context_size?: Exclude<NativeWebSearchContextSize, 'omit'>
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Whether the current model-visible Harness catalog exposes web_search. */
export function hasHarnessWebSearch(tools: readonly Pick<Tool, 'name'>[] | undefined): boolean {
  return tools?.some(tool => tool.name === 'web_search') === true
}

/** Build the hosted tool shape emitted by current Codex Responses clients. */
export function hostedNativeWebSearchTool(
  mode: NativeWebSearchMode,
  contextSize: NativeWebSearchContextSize,
): HostedNativeWebSearchTool {
  let access: Pick<HostedNativeWebSearchTool, 'external_web_access' | 'indexed_web_access'>
  switch (mode) {
    case 'cached':
      access = { external_web_access: false }
      break
    case 'indexed':
      access = { external_web_access: true, indexed_web_access: true }
      break
    case 'live':
      access = { external_web_access: true }
      break
    default:
      throw new TypeError(`unsupported native web search mode ${JSON.stringify(mode)}`)
  }
  if (!(NATIVE_WEB_SEARCH_CONTEXT_SIZES as readonly unknown[]).includes(contextSize)) {
    throw new TypeError(`unsupported native web search context size ${JSON.stringify(contextSize)}`)
  }
  return {
    type: 'web_search',
    ...access,
    ...contextSize === 'omit' ? {} : { search_context_size: contextSize },
  }
}

/**
 * Rewrite one final Codex Responses payload without mutating the input.
 * `undefined` means that native search is not authorized for this request.
 */
export function transformNativeWebSearchPayload(
  payload: unknown,
  options: NativeWebSearchTransformOptions,
): JsonRecord | undefined {
  if (!options.enabled) return undefined
  if (!options.alwaysAvailable && !hasHarnessWebSearch(options.harnessTools)) return undefined
  if (!isRecord(payload)) return undefined

  const hosted = hostedNativeWebSearchTool(options.mode, options.contextSize)
  const tools = Array.isArray(payload['tools']) ? payload['tools'] : []
  const nextTools: unknown[] = []
  let hostedInserted = false

  for (const tool of tools) {
    if (isRecord(tool) && tool['type'] === 'function' && tool['name'] === 'web_search') continue
    if (isRecord(tool) && tool['type'] === 'web_search') {
      if (!hostedInserted) {
        nextTools.push(hosted)
        hostedInserted = true
      }
      continue
    }
    nextTools.push(tool)
  }
  if (!hostedInserted) nextTools.push(hosted)

  const include = Array.isArray(payload['include']) ? payload['include'] : []
  const nextInclude = include.includes('web_search_call.action.sources')
    ? include
    : [...include, 'web_search_call.action.sources']
  return { ...payload, tools: nextTools, include: nextInclude }
}
