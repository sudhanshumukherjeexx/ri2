import { Link } from 'react-router-dom'
import { PAGES, type PageId } from '../pagesConfig'

export function Pager({ current }: { current: PageId }) {
  const index = PAGES.findIndex((p) => p.id === current)
  const prev = index > 0 ? PAGES[index - 1] : null
  const next = index < PAGES.length - 1 ? PAGES[index + 1] : null

  if (!prev && !next) return null

  return (
    <nav className="pager" aria-label="Pipeline navigation">
      {prev ? (
        <Link to={`/ch/${prev.slug}`} className="pager-card pager-card-prev">
          <span className="pager-dir">&larr; Previous stage</span>
          <span className="pager-label">{prev.label}</span>
        </Link>
      ) : (
        <span />
      )}
      {next ? (
        <Link to={`/ch/${next.slug}`} className="pager-card pager-card-next">
          <span className="pager-dir">Next stage &rarr;</span>
          <span className="pager-label">{next.label}</span>
        </Link>
      ) : (
        <span />
      )}
    </nav>
  )
}
