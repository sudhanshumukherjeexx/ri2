import { useEffect, useState } from 'react'
import { useData } from '../state/DataContext'
import { DataSourceSelector } from '../components/DataSourceSelector'
import { DataTable } from '../components/DataTable'
import { CodeBlock } from '../components/CodeBlock'
import { Pager } from '../components/Pager'
import { callPy } from '../workers/pyodideClient'
import type { Preview } from '../state/DataContext'

const METHODS: { value: string; label: string; description: string }[] = [
  { value: 'label', label: 'Label Encoding', description: 'Assigns each category an integer (0..n-1). Best for ordinal data.' },
  { value: 'onehot', label: 'One Hot Encoding', description: 'Creates a binary column per category. Best for nominal (unordered) data.' },
  { value: 'ordinal', label: 'Ordinal Encoding', description: 'Similar to label encoding, for data with a meaningful rank.' },
]

const METHOD_SNIPPETS: Record<string, (cols: string[]) => string> = {
  label: (cols) =>
    `from sklearn.preprocessing import LabelEncoder\nfor col in ${JSON.stringify(cols)}:\n    df[col] = LabelEncoder().fit_transform(df[col])`,
  onehot: (cols) => `df = pd.get_dummies(df, columns=${JSON.stringify(cols)}, dtype=float)`,
  ordinal: (cols) =>
    `from sklearn.preprocessing import OrdinalEncoder\ncols = ${JSON.stringify(cols)}\ndf[cols] = OrdinalEncoder().fit_transform(df[cols])`,
}

type EncodeResult = { preview: Preview; original_columns: string[]; encoded_columns: string[] }

export default function Encoding() {
  const { previews } = useData()
  const [sourceKey, setSourceKey] = useState('df')
  const [selectedColumns, setSelectedColumns] = useState<string[]>([])
  const [method, setMethod] = useState('label')
  const [result, setResult] = useState<EncodeResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const source = previews[sourceKey]

  useEffect(() => {
    setSelectedColumns([])
    setResult(null)
    setError(null)
  }, [sourceKey])

  function toggleColumn(col: string) {
    setSelectedColumns((cols) =>
      cols.includes(col) ? cols.filter((c) => c !== col) : [...cols, col],
    )
  }

  async function apply() {
    setBusy(true)
    setError(null)
    try {
      const res = await callPy<EncodeResult>('encoding_apply', {
        source_key: sourceKey,
        columns: selectedColumns,
        method,
      })
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <h1>Feature Encoding</h1>
      <p className="page-intro">
        Convert categorical columns into numeric form for machine learning. Results are written
        to a new <code>Encoded</code> dataset.
      </p>

      <DataSourceSelector value={sourceKey} onChange={setSourceKey} />

      {!source && <p className="status">Load a dataset first.</p>}

      {source && (
        <>
          <section>
            <h2>Columns to Encode</h2>
            <div className="checkbox-grid">
              {source.columns.map((col) => (
                <label key={col} className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={selectedColumns.includes(col)}
                    onChange={() => toggleColumn(col)}
                  />
                  {col} <span className="data-table-dtype">({source.dtypes[col]})</span>
                </label>
              ))}
            </div>
          </section>

          <section>
            <h2>Encoding Method</h2>
            <div className="upload-row">
              <select value={method} onChange={(e) => setMethod(e.target.value)}>
                {METHODS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
              <button type="button" disabled={busy || selectedColumns.length === 0} onClick={apply}>
                Apply Encoding
              </button>
            </div>
            <p className="status">{METHODS.find((m) => m.value === method)?.description}</p>
            {error && <p className="status status-error">{error}</p>}
          </section>

          {result && (
            <section>
              <h2>Result</h2>
              <p className="status">
                {result.original_columns.length} columns &rarr; {result.encoded_columns.length}{' '}
                columns
              </p>
              <DataTable preview={result.preview} />
              <CodeBlock filename="encoding.py" code={METHOD_SNIPPETS[method]?.(selectedColumns) ?? ''} />
            </section>
          )}
        </>
      )}

      <Pager current="encoding" />
    </div>
  )
}
