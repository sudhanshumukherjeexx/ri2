import { useEffect, useMemo, useState } from 'react'
import { useData } from '../state/DataContext'
import { DataSourceSelector } from '../components/DataSourceSelector'
import { PlotlyChart } from '../components/PlotlyChart'
import { CodeBlock } from '../components/CodeBlock'
import { Pager } from '../components/Pager'
import { callPy } from '../workers/pyodideClient'

const PX_CALL: Record<string, string> = {
  box: 'px.box',
  histogram: 'px.histogram',
  scatter: 'px.scatter',
  bar: 'px.bar',
  pie: 'px.pie',
  line: 'px.line',
  violin: 'px.violin',
  contour: 'px.density_contour',
  histcontour: 'go.Histogram2dContour',
  scatter3d: 'px.scatter_3d',
  line3d: 'go.Scatter3d',
  polarscatter: 'px.scatter_polar',
  polarbar: 'px.bar_polar',
  scattergeo: 'px.scatter_geo',
  choropleth: 'px.choropleth',
  bubblemap: 'px.scatter_geo',
}

function pxSnippet(kind: string, fields: Record<string, string>) {
  const args = Object.entries(fields)
    .filter(([, v]) => v)
    .map(([k, v]) => `${k}='${v}'`)
    .join(', ')
  return `fig = ${PX_CALL[kind] ?? 'px.scatter'}(df, ${args})`
}

type FieldSpec = {
  name: string
  label: string
  type: 'numeric' | 'categorical' | 'any' | 'choice'
  optional?: boolean
  options?: string[]
}

type ChartSpec = { label: string; category: string; fields: FieldSpec[] }

function isNumericDtype(dtype: string) {
  return /int|float/i.test(dtype)
}

export default function EDA() {
  const { previews, markComplete } = useData()
  const [sourceKey, setSourceKey] = useState('df')
  const [specs, setSpecs] = useState<Record<string, ChartSpec>>({})
  const [kind, setKind] = useState<string | null>(null)
  const [fieldValues, setFieldValues] = useState<Record<string, string>>({})
  const [figure, setFigure] = useState<{ data: unknown[]; layout: Record<string, unknown> } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const source = previews[sourceKey]

  useEffect(() => {
    callPy<Record<string, ChartSpec>>('eda_specs', {}).then(setSpecs)
  }, [])

  const categories = useMemo(() => {
    const byCategory: Record<string, { key: string; spec: ChartSpec }[]> = {}
    for (const [key, spec] of Object.entries(specs)) {
      byCategory[spec.category] ??= []
      byCategory[spec.category].push({ key, spec })
    }
    return byCategory
  }, [specs])

  const numericColumns = source ? source.columns.filter((c) => isNumericDtype(source.dtypes[c])) : []

  function selectChart(key: string) {
    setKind(key)
    setFigure(null)
    setError(null)
    setFieldValues({})
  }

  function columnsFor(type: FieldSpec['type']) {
    if (!source) return []
    if (type === 'numeric') return numericColumns
    // 'categorical' intentionally includes numeric columns too -- many real
    // datasets encode categories as 0/1/2 integers (Sex, Pclass, Survived),
    // so restricting to string dtypes would make Group by/Category/Color by
    // unusable on them (same fix applied in StatTests.tsx).
    return source.columns
  }

  async function generate() {
    if (!kind) return
    setBusy(true)
    setError(null)
    try {
      const fig = await callPy<{ data: unknown[]; layout: Record<string, unknown> }>('eda_figure', {
        source_key: sourceKey,
        kind,
        fields: fieldValues,
      })
      setFigure(fig)
      markComplete('eda')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const activeSpec = kind ? specs[kind] : null
  const requiredFieldsFilled =
    !!activeSpec && activeSpec.fields.every((f) => f.optional || fieldValues[f.name])

  return (
    <div className="page">
      <h1>Exploratory Data Analysis</h1>
      <p className="page-intro">
        Pick a chart type, choose the columns to plot, and render it &mdash; computed and drawn
        entirely in your browser.
      </p>

      <DataSourceSelector value={sourceKey} onChange={setSourceKey} />

      {!source && <p className="status">Load a dataset first.</p>}

      {source && (
        <>
          {Object.entries(categories).map(([category, items]) => (
            <section key={category}>
              <h2>{category}</h2>
              <div className="chart-grid">
                {items.map(({ key, spec }) => (
                  <button
                    key={key}
                    type="button"
                    className={kind === key ? 'chart-btn chart-btn-active' : 'chart-btn'}
                    onClick={() => selectChart(key)}
                  >
                    {spec.label}
                  </button>
                ))}
              </div>
            </section>
          ))}

          {activeSpec && (
            <section>
              <h2>{activeSpec.label}</h2>
              <div className="upload-row">
                {activeSpec.fields.map((f) => (
                  <label key={f.name}>
                    {f.label}
                    {f.optional ? ' (optional)' : ''}:{' '}
                    <select
                      value={fieldValues[f.name] ?? ''}
                      onChange={(e) => setFieldValues((v) => ({ ...v, [f.name]: e.target.value }))}
                    >
                      <option value="">{f.optional ? 'None' : 'Select...'}</option>
                      {(f.type === 'choice' ? f.options ?? [] : columnsFor(f.type)).map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
                <button type="button" disabled={busy || !requiredFieldsFilled} onClick={generate}>
                  Generate
                </button>
              </div>
              {error && <p className="status status-error">{error}</p>}
              {figure && (
                <>
                  <PlotlyChart data={figure.data as never} layout={figure.layout as never} />
                  <CodeBlock filename="eda.py" code={pxSnippet(kind!, fieldValues)} />
                </>
              )}
            </section>
          )}
        </>
      )}

      <Pager current="eda" />
    </div>
  )
}
