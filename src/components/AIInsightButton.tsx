import { useState } from 'react'
import { useOpenAiKey } from '../state/useOpenAiKey'

type Props = {
  /** Plain-text description of what to analyze (stats, test results, etc.) --
   * never a chart screenshot, since headless image capture isn't feasible
   * client-side. */
  summary: string
  /** What kind of insight to ask for, e.g. "Explain these summary statistics". */
  prompt: string
  label?: string
}

// OpenAI's API has no Access-Control-Allow-Origin header, so a static site
// can't call it directly from the browser (CORS blocks it outright). This
// goes through a stateless forwarding proxy instead -- see proxy/README.md.
// The proxy never stores or reads the key, only relays it to OpenAI.
const PROXY_URL = import.meta.env.VITE_AI_PROXY_URL as string | undefined

async function fetchInsight(apiKey: string, prompt: string, summary: string) {
  if (!PROXY_URL) {
    throw new Error(
      'AI Insight proxy is not configured. See proxy/README.md to deploy it, then set VITE_AI_PROXY_URL.',
    )
  }

  const res = await fetch(PROXY_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content:
            'You are a concise data analyst. Explain findings in plain language, in 3-5 short sentences or bullet points. No preamble.',
        },
        { role: 'user', content: `${prompt}\n\nData:\n${summary}` },
      ],
      max_tokens: 400,
    }),
  })

  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.error?.message ?? `OpenAI request failed (${res.status})`)
  }
  const data = await res.json()
  return data.choices?.[0]?.message?.content ?? 'No response.'
}

/**
 * Every OpenAI call here runs entirely client-side with the visitor's own
 * key (stored only in localStorage) -- there is no server component to this
 * app, so no key is ever sent anywhere except directly to OpenAI.
 */
export function AIInsightButton({ summary, prompt, label = 'Get AI Insight' }: Props) {
  const { apiKey, setApiKey, clearApiKey } = useOpenAiKey()
  const [keyInput, setKeyInput] = useState('')
  const [showKeyForm, setShowKeyForm] = useState(false)
  const [insight, setInsight] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleClick() {
    if (!apiKey) {
      setShowKeyForm(true)
      return
    }
    setLoading(true)
    setError(null)
    try {
      const text = await fetchInsight(apiKey, prompt, summary)
      setInsight(text)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  function submitKey() {
    if (!keyInput.trim()) return
    setApiKey(keyInput.trim())
    setKeyInput('')
    setShowKeyForm(false)
  }

  return (
    <div className="ai-insight">
      {!showKeyForm && (
        <button type="button" onClick={handleClick} disabled={loading}>
          {loading ? 'Thinking…' : label}
        </button>
      )}

      {showKeyForm && (
        <div className="upload-row">
          <input
            type="password"
            placeholder="sk-..."
            value={keyInput}
            onChange={(e) => setKeyInput(e.target.value)}
            style={{ width: 260 }}
          />
          <button type="button" onClick={submitKey}>
            Save Key
          </button>
          <button type="button" onClick={() => setShowKeyForm(false)}>
            Cancel
          </button>
          <span className="status">
            Your OpenAI API key is stored only in this browser and sent only to OpenAI.
          </span>
        </div>
      )}

      {apiKey && !showKeyForm && (
        <button type="button" className="ai-insight-change-key" onClick={() => setShowKeyForm(true)}>
          Change key
        </button>
      )}
      {apiKey && !showKeyForm && (
        <button type="button" className="ai-insight-change-key" onClick={clearApiKey}>
          Remove key
        </button>
      )}

      {error && <p className="status status-error">{error}</p>}
      {insight && <p className="ai-insight-text">{insight}</p>}
    </div>
  )
}
