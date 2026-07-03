import { useEffect, useState } from 'react'
import { getHealth } from './api'
import PredictTab from './tabs/PredictTab'
import BatchTab from './tabs/BatchTab'
import MonitoringTab from './tabs/MonitoringTab'
import './App.css'

const TABS = [
  { id: 'predict', label: 'Predict', component: PredictTab },
  { id: 'batch', label: 'Batch', component: BatchTab },
  { id: 'monitoring', label: 'Monitoring', component: MonitoringTab },
]

function HealthBadge() {
  const [status, setStatus] = useState('checking')

  useEffect(() => {
    let cancelled = false

    getHealth()
      .then((data) => {
        if (!cancelled) setStatus(data.status === 'healthy' ? 'online' : 'degraded')
      })
      .catch(() => {
        if (!cancelled) setStatus('offline')
      })

    return () => {
      cancelled = true
    }
  }, [])

  return <span className={`badge badge-${status}`}>API: {status}</span>
}

export default function App() {
  const [activeTab, setActiveTab] = useState('predict')
  const ActiveComponent = TABS.find((tab) => tab.id === activeTab).component

  return (
    <div className="page">
      <header className="header">
        <h1 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 30, height: 30, borderRadius: 8, background: 'rgba(250,204,21,0.15)', border: '1px solid rgba(250,204,21,0.35)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#facc15" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 19v-6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2zm0 0V9a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v10m-6 0a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2m0 0V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-2a2 2 0 0 1-2-2z"/>
            </svg>
          </span>
          MoodGuess-MLOps
        </h1>
        <HealthBadge />
      </header>

      <nav className="tab-nav">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`tab-nav-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <ActiveComponent />
    </div>
  )
}
