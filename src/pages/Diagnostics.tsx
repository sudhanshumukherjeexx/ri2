import { useEffect, useState } from 'react'
import { useData } from '../state/DataContext'
import { DataSourceSelector } from '../components/DataSourceSelector'
import { PlotlyChart } from '../components/PlotlyChart'
import { AIInsightButton } from '../components/AIInsightButton'
import { CodeBlock } from '../components/CodeBlock'
import { Pager } from '../components/Pager'
import { callPy } from '../workers/pyodideClient'

type SummaryRow = { column: string; skewness: number; kurtosis: number }
type Summary = { columns: string[]; rows: SummaryRow[] }

type ColumnDiagnostics = {
  column: string
  n: number
  mean: number
  std: number
  skewness: number
  kurtosis: number
  histogram_values: number[]
  qq: { theoretical: number[]; sample: number[]; slope: number; intercept: number; r: number }
  normality_tests: {
    shapiro: { statistic: number; p_value: number }
    kolmogorov_smirnov: { statistic: number; p_value: number }
    anderson_darling: { statistic: number; critical_values: number[]; significance_levels: number[] }
  }
}

type OutlierResult = {
  column: string
  k: number
  q1: number
  q3: number
  iqr: number
  lower_bound: number
  upper_bound: number
  outlier_count: number
  total_count: number
  outlier_values: number[]
}

const TABS = ['Distribution & Q-Q', 'Skewness', 'Kurtosis', 'Outliers'] as const
type Tab = (typeof TABS)[number]

export default function Diagnostics() {
  const { previews, markComplete } = useData()
  const [sourceKey, setSourceKey] = useState('df')
  const [tab, setTab] = useState<Tab>('Distribution & Q-Q')
  const [summary, setSummary] = useState<Summary | null>(null)
  const [column, setColumn] = useState('')
  const [colDiag, setColDiag] = useState<ColumnDiagnostics | null>(null)
  const [k, setK] = useState('1.5')
  const [outliers, setOutliers] = useState<OutlierResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const source = previews[sourceKey]

  useEffect(() => {
    if (!source) return
    setBusy(true)
    setError(null)
    callPy<Summary>('diagnostics_summary', { source_key: sourceKey })
      .then((s) => {
        setSummary(s)
        setColumn(s.columns[0] ?? '')
        setColDiag(null)
        setOutliers(null)
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, source?.key])

  useEffect(() => {
    if (!column || tab !== 'Distribution & Q-Q') return
    setBusy(true)
    setError(null)
    callPy<ColumnDiagnostics>('diagnostics_column', { source_key: sourceKey, column })
      .then((d) => {
        setColDiag(d)
        markComplete('diagnostics')
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false))
  }, [column, tab, sourceKey, markComplete])

  async function detectOutliers() {
    setBusy(true)
    setError(null)
    try {
      const result = await callPy<OutlierResult>('diagnostics_outliers', {
        source_key: sourceKey,
        column,
        k: Number(k),
      })
      setOutliers(result)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page">
      <h1>Distribution Diagnostics</h1>
      <p className="page-intro">
        Understand data shape via skewness and kurtosis, test for normality, and detect outliers
        with the IQR method.
      </p>

      <DataSourceSelector value={sourceKey} onChange={setSourceKey} />

      {!source && <p className="status">Load a dataset first.</p>}

      {source && summary && (
        <>
          <div className="tab-row">
            {TABS.map((t) => (
              <button
                key={t}
                type="button"
                className={tab === t ? 'tab-btn tab-btn-active' : 'tab-btn'}
                onClick={() => setTab(t)}
              >
                {t}
              </button>
            ))}
          </div>

          {(tab === 'Distribution & Q-Q' || tab === 'Outliers') && (
            <div className="upload-row">
              <label>
                Column:{' '}
                <select value={column} onChange={(e) => setColumn(e.target.value)}>
                  {summary.columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}

          {error && <p className="status status-error">{error}</p>}

          {tab === 'Distribution & Q-Q' && colDiag && (
            <>
              <section>
                <h2>Histogram</h2>
                <PlotlyChart data={[{ type: 'histogram', x: colDiag.histogram_values }]} height={340} />
              </section>
              <section>
                <h2>Q-Q Plot (vs. Normal Distribution)</h2>
                <PlotlyChart
                  data={[
                    {
                      type: 'scatter',
                      mode: 'markers',
                      x: colDiag.qq.theoretical,
                      y: colDiag.qq.sample,
                      name: 'Sample',
                    },
                    {
                      type: 'scatter',
                      mode: 'lines',
                      x: colDiag.qq.theoretical,
                      y: colDiag.qq.theoretical.map((t) => colDiag.qq.slope * t + colDiag.qq.intercept),
                      name: 'Reference',
                      line: { color: 'red' },
                    },
                  ]}
                  layout={{ xaxis: { title: { text: 'Theoretical Quantiles' } }, yaxis: { title: { text: 'Sample Quantiles' } } }}
                  height={340}
                />
              </section>
              <section>
                <h2>Normality Tests</h2>
                <p className="status">
                  n = {colDiag.n}, mean = {colDiag.mean.toFixed(3)}, std = {colDiag.std.toFixed(3)}, skewness ={' '}
                  {colDiag.skewness.toFixed(3)}, kurtosis = {colDiag.kurtosis.toFixed(3)}
                </p>
                <ul className="missing-list">
                  <li>
                    Shapiro-Wilk: statistic = {colDiag.normality_tests.shapiro.statistic.toFixed(4)}, p ={' '}
                    {colDiag.normality_tests.shapiro.p_value.toFixed(4)}
                  </li>
                  <li>
                    Kolmogorov-Smirnov: statistic ={' '}
                    {colDiag.normality_tests.kolmogorov_smirnov.statistic.toFixed(4)}, p ={' '}
                    {colDiag.normality_tests.kolmogorov_smirnov.p_value.toFixed(4)}
                  </li>
                  <li>
                    Anderson-Darling: statistic ={' '}
                    {colDiag.normality_tests.anderson_darling.statistic.toFixed(4)} (critical values at 5%:{' '}
                    {colDiag.normality_tests.anderson_darling.critical_values[2]?.toFixed(3)})
                  </li>
                </ul>
                <AIInsightButton
                  label="Explain these results"
                  prompt={`Explain what these distribution diagnostics for column '${colDiag.column}' mean in plain language, and whether the data looks normally distributed.`}
                  summary={JSON.stringify(colDiag, null, 0)}
                />
                <CodeBlock
                  filename="diagnostics.py"
                  code={`from scipy import stats\n\nshapiro_stat, shapiro_p = stats.shapiro(df['${colDiag.column}'])\nks_stat, ks_p = stats.kstest(standardized, 'norm')\nanderson = stats.anderson(df['${colDiag.column}'], dist='norm')`}
                />
              </section>
            </>
          )}

          {(tab === 'Skewness' || tab === 'Kurtosis') && (
            <section>
              <h2>{tab} by Column</h2>
              <PlotlyChart
                data={[
                  {
                    type: 'bar',
                    x: summary.rows.map((r) => r.column),
                    y: summary.rows.map((r) => (tab === 'Skewness' ? r.skewness : r.kurtosis)),
                  },
                ]}
                height={400}
              />
            </section>
          )}

          {tab === 'Outliers' && (
            <section>
              <div className="upload-row">
                <label>
                  IQR multiplier (k):{' '}
                  <input
                    type="number"
                    step="0.1"
                    value={k}
                    onChange={(e) => setK(e.target.value)}
                    style={{ width: 70 }}
                  />
                </label>
                <button type="button" disabled={busy} onClick={detectOutliers}>
                  Detect Outliers
                </button>
              </div>

              {outliers && (
                <>
                  <p className="status">
                    Q1 = {outliers.q1.toFixed(3)}, Q3 = {outliers.q3.toFixed(3)}, IQR ={' '}
                    {outliers.iqr.toFixed(3)}, bounds = [{outliers.lower_bound.toFixed(3)},{' '}
                    {outliers.upper_bound.toFixed(3)}]
                  </p>
                  <p className="status">
                    {outliers.outlier_count} of {outliers.total_count} values (
                    {((outliers.outlier_count / outliers.total_count) * 100).toFixed(1)}%) are outliers
                  </p>
                  {outliers.outlier_values.length > 0 && (
                    <p className="status">
                      Sample: {outliers.outlier_values.slice(0, 20).map((v) => v.toFixed(2)).join(', ')}
                    </p>
                  )}
                </>
              )}
            </section>
          )}
        </>
      )}

      <Pager current="diagnostics" />
    </div>
  )
}
