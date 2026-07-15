import { useEffect, useState } from 'react'
import { useData } from '../state/DataContext'
import { DataSourceSelector } from '../components/DataSourceSelector'
import { DataTable } from '../components/DataTable'
import { PlotlyChart } from '../components/PlotlyChart'
import { CodeBlock } from '../components/CodeBlock'
import { Pager } from '../components/Pager'
import { callPy } from '../workers/pyodideClient'
import type { Preview } from '../state/DataContext'

type Comparison = { column: string; before: number[]; after: number[] }
type ScaleResult = { preview: Preview; comparisons: Comparison[] }

function isNumericDtype(dtype: string) {
  return /int|float/i.test(dtype)
}

const METHOD_SNIPPETS: Record<string, (cols: string[]) => string> = {
  minmax: (cols) => `from sklearn.preprocessing import MinMaxScaler\ndf[${JSON.stringify(cols)}] = MinMaxScaler().fit_transform(df[${JSON.stringify(cols)}])`,
  standard: (cols) => `from sklearn.preprocessing import StandardScaler\ndf[${JSON.stringify(cols)}] = StandardScaler().fit_transform(df[${JSON.stringify(cols)}])`,
  robust: (cols) => `from sklearn.preprocessing import RobustScaler\ndf[${JSON.stringify(cols)}] = RobustScaler().fit_transform(df[${JSON.stringify(cols)}])`,
  maxabs: (cols) => `from sklearn.preprocessing import MaxAbsScaler\ndf[${JSON.stringify(cols)}] = MaxAbsScaler().fit_transform(df[${JSON.stringify(cols)}])`,
  quantile: (cols) => `from sklearn.preprocessing import QuantileTransformer\ndf[${JSON.stringify(cols)}] = QuantileTransformer().fit_transform(df[${JSON.stringify(cols)}])`,
  log: (cols) => `df[${JSON.stringify(cols)}] = np.log1p(df[${JSON.stringify(cols)}])`,
  boxcox: (cols) => `from sklearn.preprocessing import PowerTransformer\ndf[${JSON.stringify(cols)}] = PowerTransformer(method='box-cox').fit_transform(df[${JSON.stringify(cols)}])`,
  yeojohnson: (cols) => `from sklearn.preprocessing import PowerTransformer\ndf[${JSON.stringify(cols)}] = PowerTransformer(method='yeo-johnson').fit_transform(df[${JSON.stringify(cols)}])`,
}

export default function Scaling() {
  const { previews } = useData()
  const [sourceKey, setSourceKey] = useState('df')
  const [methods, setMethods] = useState<Record<string, string>>({})
  const [selectedColumns, setSelectedColumns] = useState<string[]>([])
  const [method, setMethod] = useState('')
  const [result, setResult] = useState<ScaleResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const source = previews[sourceKey]
  const numericColumns = source ? source.columns.filter((c) => isNumericDtype(source.dtypes[c])) : []

  useEffect(() => {
    callPy<Record<string, string>>('scaling_methods', {}).then((m) => {
      setMethods(m)
      setMethod(Object.keys(m)[0] ?? '')
    })
  }, [])

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
      const res = await callPy<ScaleResult>('scaling_apply', {
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
      <h1>Feature Scaling and Transformation</h1>
      <p className="page-intro">
        Normalize, standardize, or transform numeric features. Results are written to a new{' '}
        <code>Scaled</code> dataset.
      </p>

      <DataSourceSelector value={sourceKey} onChange={setSourceKey} />

      {!source && <p className="status">Load a dataset first.</p>}

      {source && (
        <>
          <section>
            <h2>Columns to Scale</h2>
            {numericColumns.length === 0 ? (
              <p className="status">No numeric columns found in this dataset.</p>
            ) : (
              <div className="checkbox-grid">
                {numericColumns.map((col) => (
                  <label key={col} className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={selectedColumns.includes(col)}
                      onChange={() => toggleColumn(col)}
                    />
                    {col}
                  </label>
                ))}
              </div>
            )}
          </section>

          <section>
            <h2>Method</h2>
            <div className="upload-row">
              <select value={method} onChange={(e) => setMethod(e.target.value)}>
                {Object.entries(methods).map(([key, desc]) => (
                  <option key={key} value={key} title={desc}>
                    {key}
                  </option>
                ))}
              </select>
              <button type="button" disabled={busy || selectedColumns.length === 0} onClick={apply}>
                Apply Scaling
              </button>
            </div>
            {methods[method] && <p className="status">{methods[method]}</p>}
            {error && <p className="status status-error">{error}</p>}
          </section>

          {result && (
            <>
              <section>
                <h2>Result</h2>
                <DataTable preview={result.preview} />
                <CodeBlock filename="scaling.py" code={METHOD_SNIPPETS[method]?.(selectedColumns) ?? ''} />
              </section>

              <section>
                <h2>Before / After Distribution</h2>
                {result.comparisons.map((c) => (
                  <div key={c.column} style={{ marginBottom: 24 }}>
                    <h3>{c.column}</h3>
                    <PlotlyChart
                      data={[
                        { type: 'histogram', x: c.before, name: 'Original', opacity: 0.7 },
                        { type: 'histogram', x: c.after, name: 'Scaled', opacity: 0.7 },
                      ]}
                      layout={{ barmode: 'overlay' }}
                      height={320}
                    />
                  </div>
                ))}
              </section>
            </>
          )}
        </>
      )}

      <Pager current="scaling" />
    </div>
  )
}
