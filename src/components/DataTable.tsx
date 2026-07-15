import type { Preview } from '../state/DataContext'

export function DataTable({ preview }: { preview: Preview }) {
  return (
    <div className="data-table-wrap">
      <div className="data-table-meta">
        {preview.shape[0].toLocaleString()} rows &times; {preview.shape[1]} columns
      </div>
      <div className="data-table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {preview.columns.map((c) => (
                <th key={c}>
                  {c}
                  <div className="data-table-dtype">{preview.dtypes[c]}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.head.map((row, i) => (
              <tr key={i}>
                {preview.columns.map((c) => (
                  <td key={c}>{row[c] === null || row[c] === undefined ? '' : String(row[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
