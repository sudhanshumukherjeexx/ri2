import { useEffect, useState } from 'react'
import { useData } from '../state/DataContext'
import { DataSourceSelector } from '../components/DataSourceSelector'
import { DataTable } from '../components/DataTable'
import { CodeBlock } from '../components/CodeBlock'
import { Pager } from '../components/Pager'
import { callPy } from '../workers/pyodideClient'
import type { Preview } from '../state/DataContext'

const DATA_TYPES = ['INT', 'FLOAT', 'DATETIME', 'BOOLEAN', 'STRING']

const CONVERT_SNIPPETS: Record<string, (col: string) => string> = {
  INT: (col) => `df['${col}'] = np.trunc(df['${col}']).astype('Int64')`,
  FLOAT: (col) => `df['${col}'] = df['${col}'].astype('float64')`,
  DATETIME: (col) => `df['${col}'] = pd.to_datetime(df['${col}'], errors='coerce')`,
  BOOLEAN: (col) => `df['${col}'] = (df['${col}'] != 0).astype('boolean')`,
  STRING: (col) => `df['${col}'] = df['${col}'].astype(str).mask(df['${col}'].isna(), None)`,
}

type StartResult = { preview: Preview; duplicate_count: number; original_dtypes: Record<string, string> }
type ActionResult = { preview: Preview; duplicate_count?: number; success?: boolean; message?: string }

export default function Cleanup() {
  const { previews, setPreview } = useData()
  const [sourceKey, setSourceKey] = useState('df')
  const [tab, setTab] = useState<'duplicates' | 'convert'>('duplicates')
  const [duplicateCount, setDuplicateCount] = useState<number | null>(null)
  const [originalDtypes, setOriginalDtypes] = useState<Record<string, string>>({})
  const [column, setColumn] = useState('')
  const [newType, setNewType] = useState(DATA_TYPES[0])
  const [message, setMessage] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const source = previews[sourceKey]
  const cleaned = previews['df_cleaned']

  useEffect(() => {
    if (!source) return
    setBusy(true)
    callPy<StartResult>('cleanup_start', { source_key: sourceKey })
      .then((result) => {
        setPreview('df_cleaned', result.preview)
        setDuplicateCount(result.duplicate_count)
        setOriginalDtypes(result.original_dtypes)
        setColumn(result.preview.columns[0] ?? '')
        setMessage(null)
      })
      .finally(() => setBusy(false))
    // Only re-run when the chosen source dataset changes, not on every
    // preview update (cleaning df_cleaned itself must not restart the session).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, source?.key])

  async function removeDuplicates() {
    setBusy(true)
    try {
      const result = await callPy<ActionResult>('cleanup_remove_duplicates', {})
      setPreview('df_cleaned', result.preview)
      setDuplicateCount(result.duplicate_count ?? 0)
    } finally {
      setBusy(false)
    }
  }

  async function convertType() {
    setBusy(true)
    try {
      const result = await callPy<ActionResult>('cleanup_convert_dtype', { column, new_type: newType })
      setPreview('df_cleaned', result.preview)
      setMessage(result.message ?? null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <h1>Data Cleanup and Conversion</h1>
      <p className="page-intro">
        Remove duplicate rows and convert column data types. Results are written to a new{' '}
        <code>Cleaned</code> dataset &mdash; your original upload is left untouched.
      </p>

      <DataSourceSelector value={sourceKey} onChange={setSourceKey} />

      {!source && <p className="status">Load a dataset first.</p>}

      {source && cleaned && (
        <>
          <div className="tab-row">
            <button
              type="button"
              className={tab === 'duplicates' ? 'tab-btn tab-btn-active' : 'tab-btn'}
              onClick={() => setTab('duplicates')}
            >
              Handle Duplicates
            </button>
            <button
              type="button"
              className={tab === 'convert' ? 'tab-btn tab-btn-active' : 'tab-btn'}
              onClick={() => setTab('convert')}
            >
              Convert Data Types
            </button>
          </div>

          {tab === 'duplicates' && (
            <section>
              {duplicateCount !== null && duplicateCount > 0 && (
                <p className="status">
                  Found {duplicateCount} duplicate rows.{' '}
                  <button type="button" disabled={busy} onClick={removeDuplicates}>
                    Remove Duplicates
                  </button>
                </p>
              )}
              {duplicateCount === 0 && <p className="status">No duplicate rows found.</p>}
              <DataTable preview={cleaned} />
              <CodeBlock
                filename="cleanup.py"
                code={`duplicate_count = df.duplicated(keep=False).sum()\ndf = df.drop_duplicates()`}
              />
            </section>
          )}

          {tab === 'convert' && (
            <section>
              <div className="upload-row">
                <label>
                  Column:{' '}
                  <select value={column} onChange={(e) => setColumn(e.target.value)}>
                    {cleaned.columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  New type:{' '}
                  <select value={newType} onChange={(e) => setNewType(e.target.value)}>
                    {DATA_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </label>
                <button type="button" disabled={busy} onClick={convertType}>
                  Convert
                </button>
              </div>

              {message && <p className="status">{message}</p>}

              <div className="dtype-compare">
                <div>
                  <h3>Original</h3>
                  {Object.entries(originalDtypes).map(([c, t]) => (
                    <div key={c} className="dtype-row">
                      {c}: {t}
                    </div>
                  ))}
                </div>
                <div>
                  <h3>Current</h3>
                  {Object.entries(cleaned.dtypes).map(([c, t]) => (
                    <div key={c} className="dtype-row">
                      {c}: {t}
                    </div>
                  ))}
                </div>
              </div>

              <DataTable preview={cleaned} />
              <CodeBlock filename="cleanup.py" code={CONVERT_SNIPPETS[newType]?.(column || 'column') ?? ''} />
            </section>
          )}
        </>
      )}

      <Pager current="cleanup" />
    </div>
  )
}
