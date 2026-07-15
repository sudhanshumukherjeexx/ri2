import { useEffect, useState } from 'react'
import { useData } from '../state/DataContext'
import { DataSourceSelector } from '../components/DataSourceSelector'
import { PlotlyChart } from '../components/PlotlyChart'
import { AIInsightButton } from '../components/AIInsightButton'
import { CodeBlock } from '../components/CodeBlock'
import { Pager } from '../components/Pager'
import { callPy } from '../workers/pyodideClient'

const TEST_SNIPPETS: Record<string, (a: string, b: string) => string> = {
  ttest: (group, value) =>
    `from scipy import stats\ngroups = [g['${value}'].to_numpy() for _, g in df.groupby('${group}')]\n_, levene_p = stats.levene(*groups)\nstat, p = stats.ttest_ind(*groups, equal_var=levene_p >= 0.05)`,
  anova: (group, value) =>
    `from scipy import stats\ngroups = [g['${value}'].to_numpy() for _, g in df.groupby('${group}')]\nstat, p = stats.f_oneway(*groups)`,
  mannwhitney: (group, value) =>
    `from scipy import stats\ngroups = [g['${value}'].to_numpy() for _, g in df.groupby('${group}')]\nstat, p = stats.mannwhitneyu(*groups)`,
  kruskal: (group, value) =>
    `from scipy import stats\ngroups = [g['${value}'].to_numpy() for _, g in df.groupby('${group}')]\nstat, p = stats.kruskal(*groups)`,
  chisquare: (a, b) =>
    `from scipy import stats\ntable = pd.crosstab(df['${a}'], df['${b}'])\nstat, p, dof, expected = stats.chi2_contingency(table)`,
  wilcoxon: (a, b) => `from scipy import stats\nstat, p = stats.wilcoxon(df['${a}'], df['${b}'])`,
}

type TestSpec = { label: string; kind: 'group' | 'two_columns'; min_groups?: number; max_groups?: number | null }

type GroupResult = {
  statistic: number
  p_value: number
  groups: { group: string; n: number; mean: number; std: number }[]
  used_welch?: boolean
}
type ChiSquareResult = {
  statistic: number
  p_value: number
  dof: number
  contingency_table: { index: string[]; columns: string[]; values: number[][] }
}
type WilcoxonResult = { statistic: number; p_value: number; n_pairs: number }
type TestResult = GroupResult | ChiSquareResult | WilcoxonResult

function isNumericDtype(dtype: string) {
  return /int|float/i.test(dtype)
}

export default function StatTests() {
  const { previews, markComplete } = useData()
  const [sourceKey, setSourceKey] = useState('df')
  const [specs, setSpecs] = useState<Record<string, TestSpec>>({})
  const [test, setTest] = useState('')
  const [groupCol, setGroupCol] = useState('')
  const [valueCol, setValueCol] = useState('')
  const [colA, setColA] = useState('')
  const [colB, setColB] = useState('')
  const [visualize, setVisualize] = useState(true)
  const [figure, setFigure] = useState<{ data: unknown[]; layout: Record<string, unknown> } | null>(null)
  const [result, setResult] = useState<TestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const source = previews[sourceKey]
  const numericColumns = source ? source.columns.filter((c) => isNumericDtype(source.dtypes[c])) : []
  // Group/category columns intentionally include numeric ones too -- many
  // real datasets encode categories as 0/1/2 integers (Sex, Pclass, Fbs...),
  // so restricting to string dtypes would make these tests unusable on them.
  const groupableColumns = source ? source.columns : []

  useEffect(() => {
    callPy<Record<string, TestSpec>>('stats_test_specs', {}).then((s) => {
      setSpecs(s)
      setTest(Object.keys(s)[0] ?? '')
    })
  }, [])

  useEffect(() => {
    setResult(null)
    setError(null)
    setFigure(null)
  }, [sourceKey, test])

  const spec = specs[test]
  const isChiSquare = test === 'chisquare'

  async function run() {
    setBusy(true)
    setError(null)
    setFigure(null)
    try {
      const args: Record<string, unknown> = { source_key: sourceKey, test }
      if (spec?.kind === 'group') {
        args.group_col = groupCol
        args.value_col = valueCol
      } else {
        args.col_a = colA
        args.col_b = colB
      }
      const res = await callPy<TestResult>('stats_test_run', args)
      setResult(res)
      markComplete('stat-tests')

      if (visualize && spec?.kind === 'group') {
        const fig = await callPy<{ data: unknown[]; layout: Record<string, unknown> }>('eda_figure', {
          source_key: sourceKey,
          kind: 'box',
          fields: { y: valueCol, x: groupCol },
        })
        setFigure(fig)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const canRun =
    !!spec &&
    (spec.kind === 'group' ? !!groupCol && !!valueCol : !!colA && !!colB && colA !== colB)

  return (
    <div className="page">
      <h1>Statistical Tests</h1>
      <p className="page-intro">
        Run hypothesis tests to compare groups, check associations, or compare paired
        measurements.
      </p>

      <DataSourceSelector value={sourceKey} onChange={setSourceKey} />

      {!source && <p className="status">Load a dataset first.</p>}

      {source && (
        <>
          <section>
            <div className="upload-row">
              <label>
                Test:{' '}
                <select value={test} onChange={(e) => setTest(e.target.value)}>
                  {Object.entries(specs).map(([key, s]) => (
                    <option key={key} value={key}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </label>

              {spec?.kind === 'group' && (
                <>
                  <label>
                    Group column:{' '}
                    <select value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                      <option value="">Select...</option>
                      {groupableColumns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Value column:{' '}
                    <select value={valueCol} onChange={(e) => setValueCol(e.target.value)}>
                      <option value="">Select...</option>
                      {numericColumns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </label>
                </>
              )}

              {spec?.kind === 'two_columns' && (
                <>
                  <label>
                    Column A:{' '}
                    <select value={colA} onChange={(e) => setColA(e.target.value)}>
                      <option value="">Select...</option>
                      {(isChiSquare ? groupableColumns : numericColumns).map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Column B:{' '}
                    <select value={colB} onChange={(e) => setColB(e.target.value)}>
                      <option value="">Select...</option>
                      {(isChiSquare ? groupableColumns : numericColumns).map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </label>
                </>
              )}

              <button type="button" disabled={busy || !canRun} onClick={run}>
                Run Test
              </button>
            </div>
            {spec?.kind === 'group' && (
              <label className="checkbox-row">
                <input type="checkbox" checked={visualize} onChange={(e) => setVisualize(e.target.checked)} />
                Show box plot of the groups
              </label>
            )}
            {error && <p className="status status-error">{error}</p>}
          </section>

          {result && (
            <section>
              <h2>Results</h2>
              <p className="status">
                Statistic = {result.statistic.toFixed(4)}, p-value = {result.p_value.toFixed(4)}
                {' — '}
                {result.p_value < 0.05
                  ? 'statistically significant at α = 0.05 (reject the null hypothesis)'
                  : 'not statistically significant at α = 0.05 (fail to reject the null hypothesis)'}
              </p>

              {'groups' in result && (
                <>
                  {test === 'ttest' && (
                    <p className="status">
                      {result.used_welch
                        ? "Group variances differ (Levene's test p < 0.05) — used Welch's t-test instead of Student's."
                        : "Group variances are similar (Levene's test p ≥ 0.05) — used Student's t-test."}
                    </p>
                  )}
                  <ul className="missing-list">
                    {result.groups.map((g) => (
                      <li key={g.group}>
                        {g.group}: n = {g.n}, mean = {g.mean.toFixed(3)}, std = {g.std.toFixed(3)}
                      </li>
                    ))}
                  </ul>
                </>
              )}

              {'contingency_table' in result && (
                <div className="data-table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th />
                        {result.contingency_table.columns.map((c) => (
                          <th key={c}>{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.contingency_table.index.map((row, i) => (
                        <tr key={row}>
                          <td>
                            <strong>{row}</strong>
                          </td>
                          {result.contingency_table.values[i].map((v, j) => (
                            <td key={j}>{v}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {'n_pairs' in result && <p className="status">{result.n_pairs} paired observations used.</p>}

              {figure && <PlotlyChart data={figure.data as never} layout={figure.layout as never} />}

              <AIInsightButton
                label="Explain this result"
                prompt={`Explain in plain language what this ${specs[test]?.label ?? test} result means and what a reader should conclude from it.`}
                summary={JSON.stringify(result, null, 0)}
              />
              <CodeBlock
                filename="stats_tests.py"
                code={
                  spec?.kind === 'group'
                    ? TEST_SNIPPETS[test]?.(groupCol, valueCol) ?? ''
                    : TEST_SNIPPETS[test]?.(colA, colB) ?? ''
                }
              />
            </section>
          )}
        </>
      )}

      <Pager current="stat-tests" />
    </div>
  )
}
