import { useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import Overview from './pages/Overview'
import Cleanup from './pages/Cleanup'
import Impute from './pages/Impute'
import Encoding from './pages/Encoding'
import Scaling from './pages/Scaling'
import EDA from './pages/EDA'
import Diagnostics from './pages/Diagnostics'
import StatTests from './pages/StatTests'
import { Topbar } from './components/Topbar'
import { Sidebar } from './components/Sidebar'
import { warmupKernel } from './workers/pyodideClient'
import './App.css'

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false)

  useEffect(() => {
    warmupKernel()
  }, [])

  return (
    <div className="app-shell">
      <Topbar onMenuClick={() => setDrawerOpen((v) => !v)} />
      <div className="app-body">
        {drawerOpen && <div className="drawer-scrim" onClick={() => setDrawerOpen(false)} />}
        <Sidebar open={drawerOpen} onNavigate={() => setDrawerOpen(false)} />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<Navigate to="/ch/overview" replace />} />
            <Route path="/ch/overview" element={<Overview />} />
            <Route path="/ch/cleanup" element={<Cleanup />} />
            <Route path="/ch/impute" element={<Impute />} />
            <Route path="/ch/encoding" element={<Encoding />} />
            <Route path="/ch/scaling" element={<Scaling />} />
            <Route path="/ch/eda" element={<EDA />} />
            <Route path="/ch/diagnostics" element={<Diagnostics />} />
            <Route path="/ch/stat-tests" element={<StatTests />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default App
