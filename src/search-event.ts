/** Legacy standalone-search event identity retained for log migration tools. */

import type { OpenAICodexSearchRequestRecord } from './search.ts'

/** Event emitted by versions through 0.2.8; current versions never write it. */
export const OPENAI_CODEX_SEARCH_MODEL_REQUEST_EVENT = 'web/openai-codex-search-llm-request'

declare module '@deepseek-ai/dsh-session/types' {
  interface SessionEventMap {
    /** Exact secret-free OpenAI Codex standalone-search request. */
    'web/openai-codex-search-llm-request': OpenAICodexSearchRequestRecord
  }
}
