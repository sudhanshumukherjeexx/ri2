import { useData } from '../state/DataContext'

const LABELS: Record<string, string> = {
  df: 'Initial Dataset',
  df_cleaned: 'After Cleanup',
  df_processed: 'After Imputation',
  df_encoded: 'After Encoding',
  df_scaled: 'After Scaling',
}

const ORDER = ['df', 'df_cleaned', 'df_processed', 'df_encoded', 'df_scaled']

type Props = {
  value: string
  onChange: (key: string) => void
}

/** Shared "which version of the dataset" dropdown, reused by every
 * transform/analysis page instead of each page re-implementing it. */
export function DataSourceSelector({ value, onChange }: Props) {
  const { previews } = useData()
  const available = ORDER.filter((k) => previews[k])

  if (available.length === 0) {
    return <p className="status">Load a dataset on the Overview page to get started.</p>
  }

  return (
    <label className="data-source-selector">
      Data source:
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {available.map((k) => (
          <option key={k} value={k}>
            {LABELS[k] ?? k}
          </option>
        ))}
      </select>
    </label>
  )
}
