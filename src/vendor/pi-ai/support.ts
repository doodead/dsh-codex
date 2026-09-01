/** Small unexported pi-ai 0.82.1 helpers required by the vendored transport. */

import type { Context, ProviderEnv, Tool } from '@earendil-works/pi-ai'

export interface CombinedAbortSignal {
  signal?: AbortSignal
  cleanup: () => void
}

export function combineAbortSignals(signals: readonly (AbortSignal | undefined)[]): CombinedAbortSignal {
  const active = signals.filter((signal): signal is AbortSignal => signal !== undefined)
  if (active.length === 0) return { cleanup: () => {} }
  const only = active[0]
  if (active.length === 1 && only !== undefined) return { signal: only, cleanup: () => {} }
  const controller = new AbortController()
  const listeners: Array<{ signal: AbortSignal; listener: () => void }> = []
  const abort = (signal: AbortSignal): void => {
    if (!controller.signal.aborted) controller.abort(signal.reason)
  }
  for (const signal of active) {
    if (signal.aborted) {
      abort(signal)
      break
    }
    const listener = (): void => { abort(signal) }
    signal.addEventListener('abort', listener, { once: true })
    listeners.push({ signal, listener })
  }
  return {
    signal: controller.signal,
    cleanup: () => {
      for (const { signal, listener } of listeners) signal.removeEventListener('abort', listener)
    },
  }
}

export function splitDeferredTools(
  context: Context,
  enabled: boolean,
  normalizeName: (name: string) => string = name => name,
): { immediate: Tool[]; deferred: Map<string, Tool> } {
  const unique = new Map((context.tools ?? []).map(tool => [normalizeName(tool.name), tool]))
  if (!enabled) return { immediate: [...unique.values()], deferred: new Map() }
  const deferredNames = new Set<string>()
  const usedNames = new Set<string>()
  for (const message of context.messages) {
    if (message.role === 'assistant') {
      for (const block of message.content) {
        if (block.type === 'toolCall') usedNames.add(normalizeName(block.name))
      }
    } else if (message.role === 'toolResult') {
      for (const name of message.addedToolNames ?? []) {
        const normalized = normalizeName(name)
        if (!usedNames.has(normalized)) deferredNames.add(normalized)
      }
    }
  }
  const immediate: Tool[] = []
  const deferred = new Map<string, Tool>()
  for (const [name, tool] of unique) {
    if (deferredNames.has(name)) deferred.set(name, tool)
    else immediate.push(tool)
  }
  return { immediate, deferred }
}

interface NormalizedProviderError {
  status?: number
  body?: string
  message: string
  messageCarriesBody: boolean
}

function safeJson(value: unknown): string {
  try {
    const serialized = JSON.stringify(value)
    return serialized === undefined ? String(value) : serialized
  } catch {
    return String(value)
  }
}

export function normalizeProviderError(error: unknown): NormalizedProviderError {
  if (!(error instanceof Error)) return { message: safeJson(error), messageCarriesBody: false }
  const value = error as Error & {
    status?: unknown
    statusCode?: unknown
    body?: unknown
    error?: unknown
    $metadata?: { httpStatusCode?: unknown }
    $response?: { statusCode?: unknown; body?: unknown }
  }
  const status = typeof value.statusCode === 'number'
    ? value.statusCode
    : typeof value.status === 'number'
      ? value.status
      : typeof value.$metadata?.httpStatusCode === 'number'
        ? value.$metadata.httpStatusCode
        : typeof value.$response?.statusCode === 'number'
          ? value.$response.statusCode
          : undefined
  const candidate = typeof value.body === 'string'
    ? value.body
    : isNonEmptyObject(value.error)
      ? safeJson(value.error)
      : typeof value.$response?.body === 'string'
        ? value.$response.body
        : isNonEmptyObject(value.$response?.body) && !isReadableStreamLike(value.$response?.body)
          ? safeJson(value.$response?.body)
          : undefined
  const trimmed = candidate?.trim()
  const body = trimmed === undefined || trimmed.length === 0 ? undefined : trimmed.slice(0, 4000)
  return {
    ...status === undefined ? {} : { status },
    ...body === undefined || body.length === 0 ? {} : { body },
    message: error.message,
    messageCarriesBody: body === undefined || error.message.includes(body),
  }
}

function isNonEmptyObject(value: unknown): boolean {
  return typeof value === 'object' && value !== null && Object.keys(value).length > 0
}

function isReadableStreamLike(value: unknown): boolean {
  return typeof value === 'object' && value !== null
    && 'pipe' in value && typeof value.pipe === 'function'
}

export function formatProviderError(error: NormalizedProviderError): string {
  if (error.messageCarriesBody || error.status === undefined || error.body === undefined) return error.message
  return `${error.status}: ${error.body}`
}

export function headersToRecord(headers: Headers): Record<string, string> {
  const result: Record<string, string> = {}
  headers.forEach((value, key) => { result[key] = value })
  return result
}

const DEFAULT_PROXY_PORTS: Record<string, number> = {
  ftp: 21, gopher: 70, http: 80, https: 443, ws: 80, wss: 443,
}

let processEnvironment: Map<string, string> | undefined

function bunSandboxEnvironment(name: string): string | undefined {
  if (typeof process === 'undefined' || !process.versions?.bun || Object.keys(process.env).length > 0) {
    return undefined
  }
  if (processEnvironment === undefined) {
    processEnvironment = new Map()
    try {
      const { readFileSync } = require('node:fs') as {
        readFileSync(path: string, encoding: BufferEncoding): string
      }
      for (const entry of readFileSync('/proc/self/environ', 'utf-8').split('\0')) {
        const separator = entry.indexOf('=')
        if (separator > 0) processEnvironment.set(entry.slice(0, separator), entry.slice(separator + 1))
      }
    } catch {}
  }
  return processEnvironment.get(name)
}

function environment(name: string): string | undefined {
  return (typeof process === 'undefined' ? undefined : process.env[name])
    || bunSandboxEnvironment(name)
}

function proxyEnv(key: string, env?: ProviderEnv): string {
  const lower = key.toLowerCase()
  const upper = key.toUpperCase()
  return env?.[lower] || env?.[upper] || environment(lower) || environment(upper) || ''
}

function shouldProxy(hostname: string, port: number, env?: ProviderEnv): boolean {
  const noProxy = proxyEnv('no_proxy', env).toLowerCase()
  if (noProxy === '') return true
  if (noProxy === '*') return false
  return noProxy.split(/[,\s]/).every(entry => {
    if (entry === '') return true
    const match = /^(.*):(\d+)$/.exec(entry)
    let host = match?.[1] ?? entry
    const requiredPort = match?.[2] === undefined ? 0 : Number.parseInt(match[2], 10)
    if (requiredPort !== 0 && requiredPort !== port) return true
    if (!/^[.*]/.test(host)) return hostname !== host
    if (host.startsWith('*')) host = host.slice(1)
    return !hostname.endsWith(host)
  })
}

export function resolveHttpProxyUrlForTarget(target: string | URL, env?: ProviderEnv): URL | undefined {
  let parsed: URL
  try { parsed = target instanceof URL ? target : new URL(target) } catch { return undefined }
  const protocol = parsed.protocol.slice(0, -1)
  const hostname = parsed.hostname
  const port = Number.parseInt(parsed.port, 10) || DEFAULT_PROXY_PORTS[protocol] || 0
  if (!shouldProxy(hostname, port, env)) return undefined
  let proxy = proxyEnv(`${protocol}_proxy`, env) || proxyEnv('all_proxy', env)
  if (proxy === '') return undefined
  if (!proxy.includes('://')) proxy = `${protocol}://${proxy}`
  let url: URL
  try {
    url = new URL(proxy)
  } catch (error: unknown) {
    throw new Error(`Invalid proxy URL ${JSON.stringify(proxy)}: ${error instanceof Error ? error.message : String(error)}`)
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL. Got ${url.protocol}`)
  }
  return url
}
