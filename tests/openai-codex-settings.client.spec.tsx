// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { OpenAICodexSettings } from '../src/client/OpenAICodexSettings.tsx'
import { en } from '../src/client/locales.ts'
import type { OpenAICodexSettingsKey } from '../src/client/locales.ts'

function t(key: OpenAICodexSettingsKey): string {
  return en[key]
}

function json(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  })
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('OpenAI Codex settings model catalog', () => {
  it('renders checkboxes and persists the provider-ordered visible subset', async () => {
    const availableModels = [
      { id: 'gpt-5.6-luna', name: 'GPT-5.6 Luna' },
      { id: 'gpt-5.6-sol', name: 'GPT-5.6 Sol' },
    ]
    let selected = availableModels.map(model => model.id)
    const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
      const path = String(input)
      if (path.endsWith('/auth/status')) return json({ status: 'signed-out' })
      if (path.endsWith('/image-tools')) return json({ modifyReadImage: true, shareImagegenWithOtherModels: true })
      if (path.endsWith('/response-api')) return json({
        useWebSocketContextReuse: false,
        useNativeCompaction: false,
        nativeWebSearch: true,
        nativeWebSearchMode: 'cached',
        nativeWebSearchContextSize: 'omit',
        nativeWebSearchAlwaysAvailable: false,
      })
      if (path.endsWith('/models')) {
        if (init?.method === 'POST') selected = (JSON.parse(String(init.body)) as { models: string[] }).models
        return json({ availableModels, models: selected })
      }
      throw new Error(`unexpected settings request: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<OpenAICodexSettings t={t} />)
    const luna = await screen.findByRole<HTMLInputElement>('checkbox', { name: /GPT-5\.6 Luna/u })
    const sol = screen.getByRole<HTMLInputElement>('checkbox', { name: /GPT-5\.6 Sol/u })
    expect(luna.checked).toBe(true)
    expect(sol.checked).toBe(true)

    fireEvent.click(luna)
    await waitFor(() => { expect(luna.checked).toBe(false) })
    const modelPost = fetchMock.mock.calls.find(([input, init]) => String(input).endsWith('/models') && init?.method === 'POST')
    expect(modelPost).toBeDefined()
    expect(JSON.parse(String(modelPost?.[1]?.body))).toEqual({ models: ['gpt-5.6-sol'] })
  })

  it('renders native search controls and persists exact wire options', async () => {
    const posts: unknown[] = []
    const responseApi = {
      useWebSocketContextReuse: false,
      useNativeCompaction: false,
      nativeWebSearch: true,
      nativeWebSearchMode: 'cached',
      nativeWebSearchContextSize: 'omit',
      nativeWebSearchAlwaysAvailable: false,
    }
    const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
      const path = String(input)
      if (path.endsWith('/auth/status')) return json({ status: 'signed-out' })
      if (path.endsWith('/image-tools')) return json({ modifyReadImage: true, shareImagegenWithOtherModels: true })
      if (path.endsWith('/models')) return json({ availableModels: [], models: [] })
      if (path.endsWith('/response-api')) {
        if (init?.method === 'POST') {
          const patch = JSON.parse(String(init.body)) as Record<string, unknown>
          posts.push(patch)
          Object.assign(responseApi, patch)
        }
        return json(responseApi)
      }
      throw new Error(`unexpected settings request: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<OpenAICodexSettings t={t} />)
    const mode = await screen.findByLabelText<HTMLSelectElement>(en.nativeWebSearchMode)
    const contextSize = screen.getByLabelText<HTMLSelectElement>(en.nativeWebSearchContextSize)
    expect(mode.value).toBe('cached')
    expect(contextSize.value).toBe('omit')
    expect([...mode.options].map(option => option.value)).toEqual(['cached', 'indexed', 'live'])
    expect([...contextSize.options].map(option => option.value)).toEqual(['omit', 'low', 'medium', 'high'])

    fireEvent.change(mode, { target: { value: 'live' } })
    fireEvent.change(contextSize, { target: { value: 'high' } })
    await waitFor(() => {
      expect(posts).toContainEqual({ nativeWebSearchMode: 'live' })
      expect(posts).toContainEqual({ nativeWebSearchContextSize: 'high' })
    })
  })
})
