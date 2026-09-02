/** OpenAI Codex adapter assembled from public dsh-llm-pi-ai extension points. */

import { createModels, defaultProviderAuthContext, getSupportedThinkingLevels } from '@earendil-works/pi-ai'
import type {
  Api, Context as PiContext, Model, Models, ModelThinkingLevel, MutableModels,
  Provider, SimpleStreamOptions,
} from '@earendil-works/pi-ai'
import { openaiCodexProvider } from '@earendil-works/pi-ai/providers/openai-codex'
import { resolveRetryPolicy } from '@deepseek-ai/dsh-llm'
import { attributionHeaders, contentHasImage, LlmError } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, ReasoningEffortId, StreamChunk } from '@deepseek-ai/dsh-llm'
import { PiAiAdapter } from '@deepseek-ai/dsh-llm-pi-ai'
import type { ResolvedPiAiProviderProfile } from '@deepseek-ai/dsh-llm-pi-ai'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import type { OpenAICodexCredentialStore } from './store.ts'
import { OPENAI_CODEX_PROVIDER } from './store.ts'
import { OpenAICodexResponseRuntime } from './responses.ts'
import type { ModelCatalogEntry, ResponseApiPreferences } from './tool-policy.ts'
import type { FastModeRegistry } from './fast-mode.ts'
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import { toPiContext } from './codex-context.ts'
import { toStreamChunks } from './codex-stream.ts'
import type { CodexStructuredBlock } from './structured-search.ts'
import {
  stream as structuredCodexStream,
  streamSimple as structuredCodexStreamSimple,
} from './vendor/pi-ai/openai-codex-responses.ts'

/** Return a detached copy of the complete pi-ai Codex model catalog. */
export function openAICodexModelCatalog(): readonly ModelCatalogEntry[] {
  return openaiCodexProvider().getModels().map(model => ({ id: model.id, name: model.name }))
}

/** Provider idle ceiling used by the composite route. */
export const OPENAI_CODEX_STREAM_IDLE_TIMEOUT_MS = 300_000

// Match dsh-llm-pi-ai's rc.2 request-image defaults while keeping the profile
// object loadable by the older rc.7 adapter, which ignores these extra fields.
const OPENAI_CODEX_MAX_REQUEST_IMAGE_BYTES = 20 * 1024 * 1024
const OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET = 2048 * 2048
const OPENAI_CODEX_REQUEST_IMAGE_MAX_BYTES = 1024 * 1024

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

/** Lift the pre-rc.7 pi-ai replay shape into the current envelope on read. */
export function migrateLegacyOpenAICodexReplayState(value: unknown): unknown {
  const legacy = record(value)
  if (legacy?.['kind'] !== 'pi-ai' || legacy['version'] !== 1 || !Array.isArray(legacy['blocks'])) return value
  const {
    blocks,
    kind: _kind,
    version: _version,
    ...response
  } = legacy
  return {
    response: { ...response, kind: 'pi-ai', version: 2 },
    blocks,
  }
}

function migrateReplayHistory(options: GenerateOptions): GenerateOptions {
  let changed = false
  const messages = options.messages.map(message => {
    if (message.source.kind !== 'model' || message.source.replayState === undefined) return message
    const replayState = migrateLegacyOpenAICodexReplayState(message.source.replayState)
    if (replayState === message.source.replayState) return message
    changed = true
    return {
      ...message,
      source: { ...message.source, replayState },
    }
  })
  return changed ? { ...options, messages } : options
}

/**
 * Codex traffic rides on chatgpt.com, which is frequently reached through a
 * local proxy tunnel that blips for tens of seconds at a time. The dsh
 * default stops after 2 retries and caps scheduled delays at 10 seconds, so
 * this provider retries longer and backs off further to ride out such a blip.
 */
export const OPENAI_CODEX_RETRY_POLICY = resolveRetryPolicy({
  mode: 'normal',
  maxRetries: 5,
  backoff: { initialDelayMs: 1_000, maxDelayMs: 30_000, jitterRatio: 0.2 },
}, 'dsh-openai-codex retryPolicy')

/**
 * Give the generic dsh adapter a request-scoped bearer-token entry without
 * changing the provider's user-facing OAuth flow. The resolver accepts only
 * the explicit override supplied by this plugin; it never discovers an API
 * key from the environment or persistent api-key credentials.
 */
function isPayloadRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Add the request-scoped Fast Mode hint without changing other payload fields. */
export function withOpenAICodexFastMode(
  provider: Provider,
  fastMode: FastModeRegistry | undefined,
): Provider {
  const streamSimple = provider.streamSimple
  return {
    ...provider,
    streamSimple(model, context: PiContext, options?: SimpleStreamOptions) {
      const enabled = provider.id === OPENAI_CODEX_PROVIDER
        && model.provider === OPENAI_CODEX_PROVIDER
        && fastMode?.isEnabled(options?.sessionId) === true
      if (!enabled) return streamSimple.call(provider, model, context, options)
      const previousOnPayload = options?.onPayload
      return streamSimple.call(provider, model, context, {
        ...options,
        async onPayload(payload, payloadModel) {
          const replaced = await previousOnPayload?.(payload, payloadModel)
          const nextPayload = replaced === undefined ? payload : replaced
          return isPayloadRecord(nextPayload)
            ? { ...nextPayload, service_tier: 'priority' }
            : nextPayload
        },
      })
    },
  }
}

function requestProvider(provider: Provider, fastMode?: FastModeRegistry): Provider {
  return {
    ...withOpenAICodexFastMode(provider, fastMode),
    auth: {
      ...provider.auth,
      apiKey: {
        name: 'OpenAI Codex OAuth bearer token',
        async resolve({ credential }) {
          const apiKey = credential?.key
          return apiKey === undefined || apiKey.length === 0
            ? undefined
            : { auth: { apiKey }, source: 'OAuth' }
        },
      },
    },
  }
}

function structuredProvider(provider: Provider): Provider {
  return {
    ...provider,
    stream: structuredCodexStream as Provider['stream'],
    streamSimple: structuredCodexStreamSimple as Provider['streamSimple'],
  }
}

function resolveReasoningLevel(
  model: Model<Api>,
  effort: ReasoningEffortId | ModelThinkingLevel | undefined,
): ModelThinkingLevel | undefined {
  if (effort === undefined) return undefined
  if (getSupportedThinkingLevels(model).some(level => level === effort)) return effort as ModelThinkingLevel
  throw new LlmError(
    `pi-ai provider "${model.provider}" model "${model.id}" does not support reasoning effort "${effort}"`,
    'UNSUPPORTED_REASONING_EFFORT',
  )
}

function requestHeaders(): Record<string, string> {
  return attributionHeaders()
}

/** Preserve Harness call purpose until the generic pi-ai adapter reaches the provider. */
class OpenAICodexAdapter extends PiAiAdapter {
  constructor(
    options: ConstructorParameters<typeof PiAiAdapter>[0],
    private readonly responses: OpenAICodexResponseRuntime,
    private readonly models: Models,
    private readonly provider: Provider,
    private readonly resolveAttachments: () => AttachmentStore | undefined,
    private readonly visibleModelIds?: () => readonly string[],
    private readonly recordStructured?: (block: CodexStructuredBlock) => void,
  ) {
    super(options)
  }

  override async listModels(provider: string) {
    const models = await super.listModels(provider)
    const visibleModelIds = this.visibleModelIds?.()
    if (visibleModelIds === undefined) return models
    const visible = new Set(visibleModelIds)
    return models.filter(model => visible.has(model.id))
  }

  /**
   * Keep DSH's prepared-call path on this adapter's structured Codex stream.
   * Newer dsh-llm-pi-ai releases bind their own snapshot stream directly from
   * prepareCall(), which would otherwise bypass this subclass override.
   * This route's profile and provider are immutable for the adapter lifetime.
   */
  override async prepareCall(provider: string, model: string, signal?: AbortSignal) {
    const prepared = await super.prepareCall(provider, model, signal)
    return {
      model: prepared.model,
      stream: (options: GenerateOptions) => this.stream(options),
    }
  }

  override async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const release = options.purpose === 'compaction'
      ? this.responses.enterCompaction(options.sessionId === undefined ? undefined : String(options.sessionId))
      : undefined
    try {
      const migrated = migrateReplayHistory(options)
      if (migrated.stop !== undefined) {
        throw new LlmError('dsh-codex-experiment does not support GenerateOptions.stop', 'UNSUPPORTED_OPTION')
      }
      const model = this.models.getModel(migrated.provider, migrated.model)
      if (model === undefined) {
        throw new LlmError(`OpenAI Codex has no configured model "${migrated.model}"`, 'UNKNOWN_MODEL')
      }
      const reasoning = resolveReasoningLevel(model, migrated.reasoningEffort)
      const apiKey = (await this.models.getAuth(OPENAI_CODEX_PROVIDER))?.auth.apiKey
      const consumer = new AbortController()
      const upstream = migrated.signal === undefined
        ? consumer.signal
        : AbortSignal.any([migrated.signal, consumer.signal])
      using watchdog = idleWatchdog(upstream, OPENAI_CODEX_STREAM_IDLE_TIMEOUT_MS, 'LLM_STREAM_IDLE_TIMEOUT')
      try {
        const containsImage = migrated.messages.some(message => contentHasImage(message.content))
        if (containsImage && !model.input.includes('image')) {
          throw new LlmError(`pi-ai model "${model.id}" does not support image input`, 'UNSUPPORTED_CONTENT')
        }
        const attachments = containsImage ? this.resolveAttachments() : undefined
        if (containsImage && attachments === undefined) {
          throw new LlmError('pi-ai image input requires the durable attachment service', 'UNSUPPORTED_CONTENT')
        }
        const context = attachments === undefined
          ? toPiContext(migrated, undefined)
          : await toPiContext(migrated, attachments)
        const events = this.provider.streamSimple(model, context, {
          ...apiKey === undefined ? {} : { apiKey },
          ...reasoning === undefined || reasoning === 'off' ? {} : { reasoning },
          ...migrated.temperature === undefined ? {} : { temperature: migrated.temperature },
          ...migrated.maxTokens === undefined ? {} : { maxTokens: migrated.maxTokens },
          ...migrated.sessionId === undefined ? {} : { sessionId: String(migrated.sessionId) },
          signal: watchdog.signal,
          headers: requestHeaders(),
          maxRetries: 0,
        })
        const iterator = toStreamChunks(
          events as never,
          model.contextWindow,
          this.recordStructured,
        )[Symbol.asyncIterator]()
        let exhausted = false
        try {
          while (true) {
            const result = await watchdog.next(iterator)
            const timeout = timeoutOf(watchdog.signal, 'LLM_STREAM_IDLE_TIMEOUT')
            if (timeout !== undefined) throw timeout
            if (result.done) {
              exhausted = true
              return
            }
            yield result.value
          }
        } finally {
          if (!exhausted) {
            consumer.abort('Codex stream consumer stopped')
            try { await iterator.return(undefined) } catch {}
          }
        }
      } catch (error: unknown) {
        if (timeoutOf(watchdog.signal, 'LLM_STREAM_IDLE_TIMEOUT') !== undefined) {
          throw new LlmError(
            `OpenAI Codex stream idle timeout after ${OPENAI_CODEX_STREAM_IDLE_TIMEOUT_MS}ms`,
            'TIMEOUT',
            { cause: error },
          )
        }
        if (migrated.signal?.aborted) throw new LlmError('OpenAI Codex request aborted by caller', 'ABORTED', { cause: error })
        throw error
      } finally {
        consumer.abort('Codex stream consumer stopped')
      }
    } finally {
      release?.()
    }
  }
}

/**
 * Create the Codex subscription adapter without requiring a dsh fork. The
 * public pi-ai adapter owns Harness message conversion, image attachment
 * resolution, streaming, and reasoning metadata. This plugin adds optional
 * Codex-native request state/compaction and supplies the provider OAuth token.
 */
export function createOpenAICodexAdapter(
  credentials: OpenAICodexCredentialStore,
  resolveAttachments: () => AttachmentStore | undefined,
  responsePreferences: () => ResponseApiPreferences,
  fastMode?: FastModeRegistry,
  visibleModelIds?: () => readonly string[],
  recordStructured?: (block: CodexStructuredBlock) => void,
): PiAiAdapter {
  const provider = structuredProvider(openaiCodexProvider())
  const responses = new OpenAICodexResponseRuntime(responsePreferences)
  const profiles = new Map<string, ResolvedPiAiProviderProfile>([[OPENAI_CODEX_PROVIDER, {
    provider: OPENAI_CODEX_PROVIDER,
    displayName: 'OpenAI Codex',
    streamIdleTimeoutMs: OPENAI_CODEX_STREAM_IDLE_TIMEOUT_MS,
    maxRequestImageBytes: OPENAI_CODEX_MAX_REQUEST_IMAGE_BYTES,
    requestImagePixelBudget: OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET,
    requestImageMaxBytes: OPENAI_CODEX_REQUEST_IMAGE_MAX_BYTES,
    retryPolicy: OPENAI_CODEX_RETRY_POLICY,
    configuredMaxTokens: new Map(),
    piProvider: responses.wrap(requestProvider(provider, fastMode)),
  }]])
  const models: MutableModels = createModels({ credentials })
  models.setProvider(provider)
  return new OpenAICodexAdapter({
    profiles: () => profiles,
    resolveApiKey: async () => (await models.getAuth(OPENAI_CODEX_PROVIDER))?.auth.apiKey,
    auth: { credentials, authContext: defaultProviderAuthContext() },
    resolveAttachments,
  }, responses, models, profiles.get(OPENAI_CODEX_PROVIDER)?.piProvider ?? provider,
  resolveAttachments, visibleModelIds, recordStructured)
}
