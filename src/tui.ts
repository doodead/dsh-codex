/** Optional dsh-tui front-door adapter for account and live preference commands. */

import { spawn } from 'node:child_process'
import type { AuthEvent, AuthPrompt } from '@earendil-works/pi-ai'
import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-commands'
import type { CommandResult } from '@deepseek-ai/dsh-commands'
import {
  NATIVE_WEB_SEARCH_CONTEXT_SIZES,
  NATIVE_WEB_SEARCH_MODES,
} from './native-web-search.ts'
import type { NativeWebSearchContextSize, NativeWebSearchMode } from './native-web-search.ts'
import type { OpenAICodexService } from './service.ts'
import type { OpenAICodexUsage } from './usage.ts'

interface TuiMarkerRuntime {}

interface TuiSubcommandNode {
  name: string
  aliases?: readonly string[]
  description: string
  descriptions?: Readonly<Partial<Record<'zh' | 'en', string>>>
  tag?: string
}

interface TuiCommandTreeRuntime {
  register(provider: {
    root: string
    descriptions?: Readonly<Partial<Record<'zh' | 'en', string>>>
    children(canonicalPath: readonly string[]): readonly TuiSubcommandNode[]
  }): () => void
}

interface CommandContext extends Context {
  openAICodex: OpenAICodexService
  commands: Context['commands']
}

interface TuiContext extends Context {
  tuiCommandTrees: TuiCommandTreeRuntime
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    /** Empty marker published while the Codex terminal adapter is active. */
    openAICodexTui: object
  }
}

export const name = 'dsh-codex-experiment-tui'
export const inject = ['openAICodex']

const HELP = [
  'Usage: /codex <status|login|logout|usage|config|set>',
  '  /codex status',
  '  /codex login',
  '  /codex logout',
  '  /codex usage',
  '  /codex config',
  '  /codex set <setting> <value>',
  '  boolean values: on | off',
  '  native-web-search-mode: cached | indexed | live',
  '  native-web-search-context: omit | low | medium | high',
].join('\n')

function translatedNode(name: string, en: string, zh: string): TuiSubcommandNode {
  return { name, description: en, descriptions: { en, zh } }
}

const CODEX_ACTIONS: readonly TuiSubcommandNode[] = [
  translatedNode('status', 'Show the ChatGPT sign-in state', '查看 ChatGPT 登录状态'),
  translatedNode('login', 'Sign in with ChatGPT in the system browser', '在系统浏览器中登录 ChatGPT'),
  translatedNode('logout', 'Remove the dsh Codex credential', '移除 dsh Codex 登录凭据'),
  translatedNode('usage', 'Show current Codex usage limits', '查看当前 Codex 用量限制'),
  translatedNode('config', 'Show live Codex settings', '查看 Codex 实时配置'),
  translatedNode('set', 'Change one live Codex setting', '修改一项 Codex 实时配置'),
]

const CODEX_SETTINGS: readonly TuiSubcommandNode[] = [
  translatedNode('read-image', 'Enhance read_image with HTTP(S) input', '为 read_image 增加 HTTP(S) 图片输入'),
  translatedNode('imagegen-other-models', 'Allow other vision models to call imagegen', '允许其他视觉模型调用 imagegen'),
  translatedNode('websocket-context', 'Reuse Codex WebSocket response context', '复用 Codex WebSocket 响应上下文'),
  translatedNode('native-compaction', 'Use Codex V2 Responses compaction', '使用 Codex V2 Responses 压缩'),
  translatedNode('native-web-search', 'Use Codex hosted web search', '使用 Codex 托管网页搜索'),
  translatedNode('native-web-search-mode', 'Choose native web search access mode', '选择原生网页搜索访问模式'),
  translatedNode('native-web-search-context', 'Choose native web search context size', '选择原生网页搜索上下文大小'),
  translatedNode('native-web-search-always-available', 'Grant native search without a Harness web_search function', '即使没有 Harness web_search 函数也授予原生搜索能力'),
]

const BOOLEAN_VALUES: readonly TuiSubcommandNode[] = [
  translatedNode('on', 'Enable this setting', '启用此设置'),
  translatedNode('off', 'Disable this setting', '关闭此设置'),
]

const NATIVE_WEB_SEARCH_MODE_VALUES = NATIVE_WEB_SEARCH_MODES.map(value => translatedNode(value, value, value))
const NATIVE_WEB_SEARCH_CONTEXT_VALUES = NATIVE_WEB_SEARCH_CONTEXT_SIZES.map(value => translatedNode(value, value, value))

const BOOLEAN_SETTINGS = new Set([
  'read-image',
  'imagegen-other-models',
  'websocket-context',
  'native-compaction',
  'native-web-search',
  'native-web-search-always-available',
])

function codexSubcommands(path: readonly string[]): readonly TuiSubcommandNode[] {
  if (path.length === 1 && path[0] === 'codex') return CODEX_ACTIONS
  if (path.length === 2 && path[0] === 'codex' && path[1] === 'set') return CODEX_SETTINGS
  if (path.length === 3 && path[0] === 'codex' && path[1] === 'set') {
    if (BOOLEAN_SETTINGS.has(path[2] ?? '')) return BOOLEAN_VALUES
    if (path[2] === 'native-web-search-mode') return NATIVE_WEB_SEARCH_MODE_VALUES
    if (path[2] === 'native-web-search-context') return NATIVE_WEB_SEARCH_CONTEXT_VALUES
  }
  return []
}

function success(text: string): CommandResult {
  return { kind: 'success', text }
}

function failure(text: string): CommandResult {
  return { kind: 'error', text }
}

function safeMessage(error: unknown): string {
  return (error instanceof Error ? error.message : String(error))
    .replace(/\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b/gu, '[redacted token]')
    .replace(/(\b(?:code|token|refresh_token|access_token)=)[^&\s]+/giu, '$1[redacted]')
    .slice(0, 1000)
}

function waitForPromptAbort(prompt: AuthPrompt): Promise<string> {
  const signal = prompt.signal
  if (signal === undefined) return new Promise<string>(() => {})
  if (signal.aborted) return Promise.reject(signal.reason)
  return new Promise<string>((_resolve, reject) => {
    signal.addEventListener('abort', () => { reject(signal.reason) }, { once: true })
  })
}

/** Open one provider-issued HTTPS challenge without passing it through shell parsing. */
function openBrowser(rawUrl: string): boolean {
  const url = new URL(rawUrl)
  if (url.protocol !== 'https:') throw new Error(`refusing to open non-HTTPS authorization URL from ${url.host}`)
  if (process.platform === 'linux' && process.env.DISPLAY === undefined && process.env.WAYLAND_DISPLAY === undefined) {
    return false
  }
  const command = process.platform === 'win32'
    ? { file: 'rundll32.exe', args: ['url.dll,FileProtocolHandler', url.href] }
    : process.platform === 'darwin'
      ? { file: 'open', args: [url.href] }
      : { file: 'xdg-open', args: [url.href] }
  const child = spawn(command.file, command.args, {
    detached: true,
    stdio: 'ignore',
    windowsHide: true,
  })
  child.on('error', () => {})
  child.unref()
  return true
}

type LoginState =
  | { status: 'idle' }
  | { status: 'signing-in' }
  | { status: 'error'; message: string }

/** Own the browser challenge while the command returns control to the TUI immediately. */
class TuiLoginController {
  private state: LoginState = { status: 'idle' }
  private operation: Promise<void> | undefined
  private cancellation: AbortController | undefined
  private challenge: Promise<string> | undefined
  private resolveChallenge: ((message: string) => void) | undefined
  private rejectChallenge: ((error: unknown) => void) | undefined

  constructor(private readonly service: OpenAICodexService) {}

  async start(): Promise<string> {
    const stored = await this.service.authStatus()
    if (stored.authenticated) return 'OpenAI Codex is already signed in.'
    if (this.operation === undefined) this.begin()
    const challenge = this.challenge
    if (challenge === undefined) throw new Error('OpenAI Codex sign-in did not create an authorization challenge')
    return await challenge
  }

  status(): LoginState {
    return this.state
  }

  async logout(): Promise<void> {
    this.cancellation?.abort(new Error('OpenAI Codex sign-in cancelled'))
    await this.operation?.catch(() => undefined)
    await this.service.logout()
    this.state = { status: 'idle' }
  }

  async dispose(): Promise<void> {
    this.cancellation?.abort(new Error('OpenAI Codex TUI adapter disposed'))
    await this.operation?.catch(() => undefined)
  }

  private begin(): void {
    const cancellation = new AbortController()
    this.cancellation = cancellation
    this.state = { status: 'signing-in' }
    this.challenge = new Promise<string>((resolve, reject) => {
      this.resolveChallenge = resolve
      this.rejectChallenge = reject
    })
    this.operation = this.service.login({
      signal: cancellation.signal,
      prompt: prompt => prompt.type === 'select'
        ? Promise.resolve('browser')
        : waitForPromptAbort(prompt),
      notify: event => { this.onEvent(event) },
    }).then(
      () => { this.state = { status: 'idle' } },
      (error: unknown) => {
        const message = safeMessage(error)
        this.state = { status: 'error', message }
        this.rejectChallenge?.(error)
      },
    ).finally(() => {
      this.operation = undefined
      this.cancellation = undefined
      this.resolveChallenge = undefined
      this.rejectChallenge = undefined
    })
  }

  private onEvent(event: AuthEvent): void {
    if (event.type !== 'auth_url') return
    try {
      const opened = openBrowser(event.url)
      this.resolveChallenge?.(opened
        ? 'Opened the ChatGPT authorization page. Use /codex status after approval.'
        : `Open this ChatGPT authorization page: ${event.url}\nUse /codex status after approval.`)
    } catch (error: unknown) {
      this.cancellation?.abort(error)
      this.rejectChallenge?.(error)
    }
  }
}

function formatExpiry(expiresAt: Date | undefined): string {
  return expiresAt === undefined || Number.isNaN(expiresAt.valueOf())
    ? ''
    : ` Access token expires ${expiresAt.toISOString()}; refresh is automatic.`
}

function formatUsage(usage: OpenAICodexUsage): string {
  const lines: string[] = []
  for (const limit of usage.rateLimits) {
    const name = limit.name ?? limit.id
    for (const window of limit.windows) {
      lines.push(`${name} (${window.windowSeconds}s): ${window.remainingPercent.toFixed(1)}% remaining`)
    }
  }
  if (usage.individualLimit !== undefined) {
    lines.push(`Individual limit: ${usage.individualLimit.remainingPercent.toFixed(1)}% remaining (${usage.individualLimit.remaining}/${usage.individualLimit.limit})`)
  }
  if (usage.credits !== undefined) {
    lines.push(`Credits: ${usage.credits.unlimited ? 'unlimited' : usage.credits.balance ?? 'available'}`)
  }
  return lines.length === 0 ? 'OpenAI Codex usage is currently unavailable.' : lines.join('\n')
}

function formatConfig(service: OpenAICodexService): string {
  const image = service.imagePreferences()
  const responses = service.responsePreferences()
  return [
    `read-image: ${image.modifyReadImage ? 'on' : 'off'}`,
    `imagegen-other-models: ${image.shareImagegenWithOtherModels ? 'on' : 'off'}`,
    `websocket-context: ${responses.useWebSocketContextReuse ? 'on' : 'off'}`,
    `native-compaction: ${responses.useNativeCompaction ? 'on' : 'off'}`,
    `native-web-search: ${responses.nativeWebSearch ? 'on' : 'off'}`,
    `native-web-search-mode: ${responses.nativeWebSearchMode}`,
    `native-web-search-context: ${responses.nativeWebSearchContextSize}`,
    `native-web-search-always-available: ${responses.nativeWebSearchAlwaysAvailable ? 'on' : 'off'}`,
  ].join('\n')
}

function booleanValue(value: string): boolean {
  if (value === 'on') return true
  if (value === 'off') return false
  throw new Error(`expected "on" or "off", received ${JSON.stringify(value)}`)
}

async function updateSetting(service: OpenAICodexService, key: string, value: string): Promise<void> {
  switch (key) {
    case 'read-image':
      await service.updateImagePreferences({ modifyReadImage: booleanValue(value) })
      return
    case 'imagegen-other-models':
      await service.updateImagePreferences({ shareImagegenWithOtherModels: booleanValue(value) })
      return
    case 'websocket-context':
      await service.updateResponsePreferences({ useWebSocketContextReuse: booleanValue(value) })
      return
    case 'native-compaction':
      await service.updateResponsePreferences({ useNativeCompaction: booleanValue(value) })
      return
    case 'native-web-search':
      await service.updateResponsePreferences({ nativeWebSearch: booleanValue(value) })
      return
    case 'native-web-search-always-available':
      await service.updateResponsePreferences({ nativeWebSearchAlwaysAvailable: booleanValue(value) })
      return
    case 'native-web-search-mode':
      if (!(NATIVE_WEB_SEARCH_MODES as readonly string[]).includes(value)) {
        throw new Error('native-web-search-mode must be cached, indexed, or live')
      }
      await service.updateResponsePreferences({ nativeWebSearchMode: value as NativeWebSearchMode })
      return
    case 'native-web-search-context':
      if (!(NATIVE_WEB_SEARCH_CONTEXT_SIZES as readonly string[]).includes(value)) {
        throw new Error('native-web-search-context must be omit, low, medium, or high')
      }
      await service.updateResponsePreferences({ nativeWebSearchContextSize: value as NativeWebSearchContextSize })
      return
    default:
      throw new Error(`unknown setting ${JSON.stringify(key)}`)
  }
}

/** Register executable commands independently from any concrete UI frontend. */
export function apply(ctx: Context): void {
  ctx.inject(['commands'], registerCodexCommand)
  ctx.inject(['tuiCommandTrees'], registerTuiCommandTree)
}

function registerCodexCommand(ctx: Context): void {
  const commandCtx = ctx as CommandContext
  const service = commandCtx.openAICodex
  const login = new TuiLoginController(service)
  const disposeCommand = commandCtx.commands.register({
    name: 'codex',
    description: 'Manage the OpenAI Codex account and provider settings',
    input: { hint: 'subcommand' },
    async handler({ rawInput }) {
      const parts = rawInput.trim().split(/\s+/u).filter(Boolean)
      const action = parts[0] ?? 'status'
      try {
        switch (action) {
          case 'status': {
            const state = login.status()
            if (state.status === 'signing-in') return success('OpenAI Codex sign-in is waiting for browser approval.')
            if (state.status === 'error') return failure(`OpenAI Codex sign-in failed: ${state.message}`)
            const status = await service.authStatus()
            return status.authenticated
              ? success(`OpenAI Codex is signed in.${formatExpiry(status.expiresAt)}`)
              : failure('OpenAI Codex is signed out. Run /codex login.')
          }
          case 'login':
            if (parts.length !== 1) return failure(HELP)
            return success(await login.start())
          case 'logout':
            if (parts.length !== 1) return failure(HELP)
            await login.logout()
            return success('OpenAI Codex is signed out.')
          case 'usage':
            if (parts.length !== 1) return failure(HELP)
            return success(formatUsage(await service.usage()))
          case 'config':
            if (parts.length !== 1) return failure(HELP)
            return success(formatConfig(service))
          case 'set': {
            if (parts.length !== 3) return failure(HELP)
            await updateSetting(service, parts[1] as string, parts[2] as string)
            return success(formatConfig(service))
          }
          default:
            return failure(HELP)
        }
      } catch (error: unknown) {
        return failure(safeMessage(error))
      }
    },
  })
  ctx.effect(() => async () => {
    disposeCommand()
    await login.dispose()
  }, 'OpenAI Codex command adapter')
}

function registerTuiCommandTree(ctx: Context): void {
  const tui = ctx as TuiContext
  const disposeTree = tui.tuiCommandTrees.register({
    root: 'codex',
    descriptions: {
      en: 'Manage the OpenAI Codex account and provider settings',
      zh: '管理 OpenAI Codex 账号与提供方设置',
    },
    children: codexSubcommands,
  })
  ctx.provide('openAICodexTui', {} as TuiMarkerRuntime)
  ctx.effect(() => disposeTree, 'OpenAI Codex TUI completion adapter')
}

export default apply
