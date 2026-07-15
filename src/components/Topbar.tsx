import { useKernelStatus } from '../workers/useKernelStatus'

const STATUS_LABEL: Record<string, string> = {
  idle: 'kernel idle',
  loading: 'booting kernel…',
  ready: 'kernel ready',
  error: 'kernel error',
}

type Props = {
  onMenuClick: () => void
}

export function Topbar({ onMenuClick }: Props) {
  const status = useKernelStatus()

  return (
    <header className="topbar">
      <button type="button" className="drawer-toggle" onClick={onMenuClick} aria-label="Toggle sidebar">
        ☰
      </button>
      <div className="topbar-brand">
        <span className="topbar-wordmark" title="Rapid Insights Data Engine">
          RI2
        </span>
        <span className="topbar-tagline">
          Analyze, visualize, and transform your data &mdash; no code, right in your browser.
        </span>
      </div>
      <div className={`kernel-status kernel-status-${status}`}>
        <span className="kernel-dot" aria-hidden="true" />
        {STATUS_LABEL[status] ?? status}
      </div>
    </header>
  )
}
