import { useCallback, useState } from 'react'

const STORAGE_KEY = 'ride_openai_api_key'

export function useOpenAiKey() {
  const [apiKey, setApiKeyState] = useState<string | null>(() =>
    localStorage.getItem(STORAGE_KEY),
  )

  const setApiKey = useCallback((key: string) => {
    localStorage.setItem(STORAGE_KEY, key)
    setApiKeyState(key)
  }, [])

  const clearApiKey = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY)
    setApiKeyState(null)
  }, [])

  return { apiKey, setApiKey, clearApiKey }
}
