import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { callPy, callPyWithFile } from '../workers/pyodideClient'
import { PAGES, type PageId } from '../pagesConfig'

export type Preview = {
  key: string
  shape: [number, number]
  columns: string[]
  dtypes: Record<string, string>
  missing: Record<string, number>
  head: Record<string, unknown>[]
  describe: Record<string, Record<string, unknown>>
}

type DataContextValue = {
  loading: boolean
  error: string | null
  previews: Record<string, Preview>
  loadFile: (file: File, key?: string) => Promise<void>
  loadSampleUrl: (url: string, filename: string, key?: string) => Promise<void>
  refreshPreview: (key: string) => Promise<void>
  setPreview: (key: string, preview: Preview) => void
  clearError: () => void
  /** True completion state, not decorative: pages tied to a derived
   * dataset (Cleanup -> df_cleaned, etc.) are complete iff that dataset
   * exists; pages without one (EDA, Diagnostics, Stat Tests) are marked
   * via markComplete() when their action succeeds. */
  isPageComplete: (id: PageId) => boolean
  markComplete: (id: PageId) => void
}

const DataContext = createContext<DataContextValue | null>(null)

export function DataProvider({ children }: { children: ReactNode }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [previews, setPreviews] = useState<Record<string, Preview>>({})
  const [completed, setCompleted] = useState<Set<PageId>>(new Set())

  const loadFile = useCallback(async (file: File, key = 'df') => {
    setLoading(true)
    setError(null)
    try {
      const preview = await callPyWithFile<Preview>('load_csv', file, { key })
      setPreviews((p) => ({ ...p, [key]: preview }))
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  const loadSampleUrl = useCallback(
    async (url: string, filename: string, key = 'df') => {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch(url)
        if (!res.ok) throw new Error(`Failed to fetch sample dataset: ${res.status}`)
        const blob = await res.blob()
        const file = new File([blob], filename)
        const preview = await callPyWithFile<Preview>('load_csv', file, { key })
        setPreviews((p) => ({ ...p, [key]: preview }))
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setLoading(false)
      }
    },
    [],
  )

  const refreshPreview = useCallback(async (key: string) => {
    const preview = await callPy<Preview>('preview', { key })
    setPreviews((p) => ({ ...p, [key]: preview }))
  }, [])

  const setPreview = useCallback((key: string, preview: Preview) => {
    setPreviews((p) => ({ ...p, [key]: preview }))
  }, [])

  const clearError = useCallback(() => setError(null), [])

  const markComplete = useCallback((id: PageId) => {
    setCompleted((prev) => (prev.has(id) ? prev : new Set(prev).add(id)))
  }, [])

  const isPageComplete = useCallback(
    (id: PageId) => {
      const page = PAGES.find((p) => p.id === id)
      if (page?.previewKey) return !!previews[page.previewKey]
      return completed.has(id)
    },
    [previews, completed],
  )

  const value = useMemo(
    () => ({
      loading,
      error,
      previews,
      loadFile,
      loadSampleUrl,
      refreshPreview,
      setPreview,
      clearError,
      isPageComplete,
      markComplete,
    }),
    [
      loading,
      error,
      previews,
      loadFile,
      loadSampleUrl,
      refreshPreview,
      setPreview,
      clearError,
      isPageComplete,
      markComplete,
    ],
  )

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>
}

export function useData() {
  const ctx = useContext(DataContext)
  if (!ctx) throw new Error('useData must be used within DataProvider')
  return ctx
}
