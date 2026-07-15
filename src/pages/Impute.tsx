import { useEffect, useState } from 'react'
import { useData } from '../state/DataContext'
import { DataSourceSelector } from '../components/DataSourceSelector'
import { DataTable } from '../components/DataTable'
import { CodeBlock } from '../components/CodeBlock'
import { Pager } from '../components/Pager'
import { callPy } from '../workers/pyodideClient'
import type { Preview } from '../state/DataContext'

type ActionResult = { preview: Preview; success: boolean; message: string }

const METHOD_SNIPPETS: Record<string, (col: string, value?: string) => string> = {
  drop: (col) => `df = df.dropna(subset=['${col}'])`,
  specific_value: (col, value) => `df['${col}'] = df['${col}'].fillna(${value ?? 'value'})`,
  ffill: (col) => `df['${col}'] = df['${col}'].ffill()`,
  bfill: (col) => `df['${col}'] = df['${col}'].bfill()`,
  distribution: (col) =>
    `mean, std = df['${col}'].mean(), df['${col}'].std()\nsampled = np.random.normal(mean, std, size=missing_count)\ndf.loc[df['${col}'].isna(), '${col}'] = np.clip(sampled, 0, None)`,
  mean: (col) => `df['${col}'] = df['${col}'].fillna(df['${col}'].mean())`,
  median: (col) => `df['${col}'] = df['${col}'].fillna(df['${col}'].median())`,
  nearest_neighbor: (col) =>
    `known = df['${col}'].dropna()\nclosest = np.abs(missing_vals - known.to_numpy()).argmin(axis=1)\ndf.loc[df['${col}'].isna(), '${col}'] = known.iloc[closest]`,
}

export default function Impute() {
  const { previews, setPreview } = useData()
  const [sourceKey, setSourceKey] = useState('df')
  const [methods, setMethods] = useState<Record<string, string>>({})
  const [column, setColumn] = useState('')
  const [method, setMethod] = useState('')
  const [value, setValue] = useState('')
  const [message, setMessage] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const source = previews[sourceKey]
  const processed = previews['df_processed']

  useEffect(() => {
    callPy<Record<string, string>>('impute_methods', {}).then((m) => {
      setMethods(m)
      setMethod(Object.keys(m)[0] ?? '')
    })
  }, [])

  useEffect(() => {
    if (!source) return
    setBusy(true)
    callPy<Preview>('impute_start', { source_key: sourceKey })
      .then((p) => {
        setPreview('df_processed', p)
        setColumn(p.columns[0] ?? '')
        setMessage(null)
      })
      .finally(() => setBusy(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, source?.key])

  async function apply() {
    setBusy(true)
    try {
      // Number('') is 0, not null/NaN -- an empty box must still reach
      // impute.py's "a replacement value is required" check, not silently
      // impute with 0.
      const specificValue = method === 'specific_value' ? (value.trim() === '' ? null : Number(value)) : null
      const result = await callPy<ActionResult>('impute_apply', {
        column,
        method,
        value: specificValue,
      })
      setPreview('df_processed', result.preview)
      setMessage(result.message)
    } finally {
      setBusy(false)
    }
  }

  const missingColumns = processed
    ? Object.entries(processed.missing).filter(([, count]) => count > 0)
    : []

  return (
    <div className="page">
      <h1>Impute Missing Values</h1>
      <p className="page-intro">
        Handle missing data with eight strategies, from simple deletion to distribution-aware
        sampling. Results are written to a new <code>Imputed</code> dataset.
      </p>

      <DataSourceSelector value={sourceKey} onChange={setSourceKey} />

      {!source && <p className="status">Load a dataset first.</p>}

      {source && processed && (
        <>
          <section>
            <h2>Missing Values Summary</h2>
            {missingColumns.length === 0 ? (
              <p>No missing values detected.</p>
            ) : (
              <ul className="missing-list">
                {missingColumns.map(([col, count]) => (
                  <li key={col}>
                    {col}: {count} missing
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section>
            <h2>Imputation Settings</h2>
            <div className="upload-row">
              <label>
                Column:{' '}
                <select value={column} onChange={(e) => setColumn(e.target.value)}>
                  {processed.columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Method:{' '}
                <select value={method} onChange={(e) => setMethod(e.target.value)}>
                  {Object.entries(methods).map(([key, desc]) => (
                    <option key={key} value={key} title={desc}>
                      {key}
                    </option>
                  ))}
                </select>
              </label>
              {method === 'specific_value' && (
                <label>
                  Value:{' '}
                  <input
                    type="number"
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    style={{ width: 100 }}
                  />
                </label>
              )}
              <button type="button" disabled={busy} onClick={apply}>
                Apply Imputation
              </button>
            </div>
            {methods[method] && <p className="status">{methods[method]}</p>}
            {message && <p className="status">{message}</p>}
          </section>

          <section>
            <h2>Result</h2>
            <DataTable preview={processed} />
            {method && column && (
              <CodeBlock filename="impute.py" code={METHOD_SNIPPETS[method]?.(column, value) ?? ''} />
            )}
          </section>
        </>
      )}

      <Pager current="impute" />
    </div>
  )
}
