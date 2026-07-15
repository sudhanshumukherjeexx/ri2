import { useEffect, useState } from 'react'
import { useData } from '../state/DataContext'
import { DataTable } from '../components/DataTable'
import { PlotlyChart } from '../components/PlotlyChart'
import { AIInsightButton } from '../components/AIInsightButton'
import { TheStalk } from '../components/TheStalk'
import { Pager } from '../components/Pager'
import { callPy } from '../workers/pyodideClient'

const SAMPLES = [
  { label: 'Iris', file: 'iris.csv' },
  { label: 'Titanic', file: 'titanic.csv' },
  { label: 'Heart Disease', file: 'heart.csv' },
  { label: 'Retail Sales', file: 'retail_sales_dataset.csv' },
]

type Correlation = { columns: string[]; matrix: number[][] }

export default function Overview() {
  const { loading, error, previews, loadFile, loadSampleUrl } = useData()
  const [correlation, setCorrelation] = useState<Correlation | null>(null)
  const preview = previews['df']

  useEffect(() => {
    if (!preview) {
      setCorrelation(null)
      return
    }
    let cancelled = false
    callPy<Correlation>('correlation', { key: 'df' }).then((result) => {
      if (!cancelled) setCorrelation(result)
    })
    return () => {
      cancelled = true
    }
  }, [preview])

  const allMissingZero = preview
    ? Object.values(preview.missing).every((v) => v === 0)
    : true

  return (
    <div className="page">
      <div className="page-header">
        <img src={`${import.meta.env.BASE_URL}brand/logo.png`} alt="RI2 logo" className="page-logo" />
        <h1>Data Overview</h1>
      </div>

      <TheStalk />

      <p className="page-intro">
        That's the mechanism in action: real pandas and scipy code, running live in your
        browser. RI2 puts it to work for you &mdash; engineer features (clean, impute,
        encode, and scale), visualize your data with 15+ chart types, and analyze it with
        statistical tests, all without writing a line of code. Upload a file or pick a bundled
        sample below to try it on real data.
      </p>

      <div className="upload-row">
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) loadFile(file, 'df')
          }}
        />
        <span className="upload-or">or try a sample:</span>
        {SAMPLES.map((s) => (
          <button
            key={s.file}
            type="button"
            onClick={() => loadSampleUrl(`${import.meta.env.BASE_URL}datasets/${s.file}`, s.file, 'df')}
          >
            {s.label}
          </button>
        ))}
      </div>
      <p className="status">Supported file formats: CSV, XLSX, XLS.</p>

      {loading && <p className="status">Loading dataset in the browser&hellip;</p>}
      {error && <p className="status status-error">{error}</p>}

      {preview && (
        <>
          <section>
            <h2>Preview</h2>
            <DataTable preview={preview} />
          </section>

          <section>
            <h2>Missing Values</h2>
            {allMissingZero ? (
              <p>No missing values detected.</p>
            ) : (
              <ul className="missing-list">
                {Object.entries(preview.missing)
                  .filter(([, count]) => count > 0)
                  .map(([col, count]) => (
                    <li key={col}>
                      {col}: {count} missing ({((count / preview.shape[0]) * 100).toFixed(1)}%)
                    </li>
                  ))}
              </ul>
            )}
          </section>

          {correlation && correlation.columns.length > 1 && (
            <section>
              <h2>Correlation Heatmap</h2>
              <PlotlyChart
                data={[
                  {
                    type: 'heatmap',
                    z: correlation.matrix,
                    x: correlation.columns,
                    y: correlation.columns,
                    colorscale: 'RdBu',
                    zmin: -1,
                    zmax: 1,
                  },
                ]}
              />
            </section>
          )}

          <section>
            <h2>AI Insight</h2>
            <AIInsightButton
              label="Explain this dataset"
              prompt="Summarize the key characteristics of this dataset and call out anything notable (skew, missing data, strong correlations)."
              summary={JSON.stringify(
                {
                  shape: preview.shape,
                  columns: preview.dtypes,
                  missing: preview.missing,
                  describe: preview.describe,
                },
                null,
                0,
              )}
            />
          </section>
        </>
      )}

      <Pager current="overview" />
    </div>
  )
}
