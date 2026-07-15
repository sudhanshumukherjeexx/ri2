import { NavLink } from 'react-router-dom'
import { PAGES } from '../pagesConfig'
import { useData } from '../state/DataContext'

type Props = {
  open: boolean
  onNavigate: () => void
}

export function Sidebar({ open, onNavigate }: Props) {
  const { isPageComplete } = useData()

  return (
    <aside className={open ? 'app-sidebar app-sidebar-open' : 'app-sidebar'}>
      <div className="sidebar-label">Pages</div>
      <nav>
        {PAGES.map((page) => (
          <NavLink
            key={page.id}
            to={`/ch/${page.slug}`}
            onClick={onNavigate}
            className={({ isActive }) => (isActive ? 'nav-link nav-link-active' : 'nav-link')}
          >
            <span
              className={isPageComplete(page.id) ? 'nav-dot nav-dot-complete' : 'nav-dot'}
              aria-hidden="true"
            />
            {page.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
