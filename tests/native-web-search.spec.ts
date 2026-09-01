import { describe, expect, it } from 'vitest'
import {
  hostedNativeWebSearchTool,
  transformNativeWebSearchPayload,
} from '../src/native-web-search.ts'

const harnessWebSearch = [{ name: 'web_search' }]

describe('Codex hosted web search payload transformation', () => {
  it('matches Codex mode fields and omits context size by default', () => {
    expect(hostedNativeWebSearchTool('cached', 'omit')).toEqual({
      type: 'web_search',
      external_web_access: false,
    })
    expect(hostedNativeWebSearchTool('live', 'omit')).toEqual({
      type: 'web_search',
      external_web_access: true,
    })
    expect(hostedNativeWebSearchTool('indexed', 'omit')).toEqual({
      type: 'web_search',
      external_web_access: true,
      indexed_web_access: true,
    })
    expect(hostedNativeWebSearchTool('cached', 'high')).toEqual({
      type: 'web_search',
      external_web_access: false,
      search_context_size: 'high',
    })
  })

  it('rejects invalid runtime mode and context values instead of failing open', () => {
    expect(() => hostedNativeWebSearchTool('invalid' as never, 'omit')).toThrow(
      'unsupported native web search mode "invalid"',
    )
    expect(() => hostedNativeWebSearchTool('cached', 'invalid' as never)).toThrow(
      'unsupported native web search context size "invalid"',
    )
  })

  it('replaces the function tool, preserves unrelated tools, and is idempotent', () => {
    const payload = {
      model: 'gpt-5.6-sol',
      tools: [
        { type: 'function', name: 'read', parameters: {} },
        { type: 'function', name: 'web_search', parameters: { type: 'object' } },
        { type: 'web_search', external_web_access: true },
        { type: 'web_search', external_web_access: true },
        { type: 'function', name: 'pwsh', parameters: {} },
      ],
    }
    const options = {
      harnessTools: harnessWebSearch,
      enabled: true,
      mode: 'cached' as const,
      contextSize: 'omit' as const,
      alwaysAvailable: false,
    }

    const once = transformNativeWebSearchPayload(payload, options)
    const twice = transformNativeWebSearchPayload(once, options)

    expect(once).toEqual(twice)
    expect(once?.['tools']).toEqual([
      { type: 'function', name: 'read', parameters: {} },
      { type: 'web_search', external_web_access: false },
      { type: 'function', name: 'pwsh', parameters: {} },
    ])
    expect(once?.['include']).toEqual(['web_search_call.action.sources'])
    expect(payload.tools).toHaveLength(5)
  })

  it('does not grant search without a model-visible permission by default', () => {
    const payload = { tools: [{ type: 'function', name: 'run_code', parameters: {} }] }
    const transformed = transformNativeWebSearchPayload(payload, {
      harnessTools: [{ name: 'run_code' }],
      enabled: true,
      mode: 'cached',
      contextSize: 'omit',
      alwaysAvailable: false,
    })

    expect(transformed).toBeUndefined()
  })

  it('can explicitly grant search without a model-visible permission', () => {
    const transformed = transformNativeWebSearchPayload({ tools: [] }, {
      harnessTools: [{ name: 'run_code' }],
      enabled: true,
      mode: 'live',
      contextSize: 'medium',
      alwaysAvailable: true,
    })

    expect(transformed?.['tools']).toEqual([{
      type: 'web_search',
      external_web_access: true,
      search_context_size: 'medium',
    }])
  })

  it('keeps the ordinary function when native search is disabled', () => {
    const payload = { tools: [{ type: 'function', name: 'web_search', parameters: {} }] }
    expect(transformNativeWebSearchPayload(payload, {
      harnessTools: harnessWebSearch,
      enabled: false,
      mode: 'cached',
      contextSize: 'omit',
      alwaysAvailable: true,
    })).toBeUndefined()
  })

  it('ignores malformed final payloads', () => {
    expect(transformNativeWebSearchPayload([], {
      harnessTools: harnessWebSearch,
      enabled: true,
      mode: 'cached',
      contextSize: 'omit',
      alwaysAvailable: false,
    })).toBeUndefined()
  })
})
