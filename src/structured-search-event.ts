/** Durable live-event bridge for structured Codex Responses records. */

import type { Context } from '@deepseek-ai/cordis'
import { KNOWN_SESSION_EVENT_TYPES } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-agent'
import {
  codexStructuredFrame,
  OPENAI_CODEX_STRUCTURED_EVENT,
} from './structured-search.ts'
import type { CodexStructuredBlock } from './structured-search.ts'

/** Register the plugin-owned event in DSH's extensible persistence vocabulary. */
export function installOpenAICodexStructuredEvent(): void {
  if (!(KNOWN_SESSION_EVENT_TYPES instanceof Set)) {
    throw new Error(
      'dsh-codex-experiment: this DSH build does not expose an extensible session event vocabulary',
    )
  }
  KNOWN_SESSION_EVENT_TYPES.add(OPENAI_CODEX_STRUCTURED_EVENT)
}

/**
 * Persist one live structured record in its initiating agent's open step.
 * Direct and maintenance LLM calls have no open step and rely on final replay
 * metadata instead.
 */
export function recordOpenAICodexStructuredBlock(
  ctx: Context,
  block: CodexStructuredBlock,
): void {
  const agent = ctx.get('agents')?.currentInitiator()
  if (agent === undefined) return
  const boundary = agent.session.events.findLast(event =>
    event.type === 'step/start' || event.type === 'step/end')
  if (boundary?.type !== 'step/start') return
  agent.session.append(OPENAI_CODEX_STRUCTURED_EVENT, {
    turn: boundary.data.turn,
    step: boundary.data.step,
    frame: codexStructuredFrame(block),
  })
}
