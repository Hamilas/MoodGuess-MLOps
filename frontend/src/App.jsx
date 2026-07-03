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
        <h1>MoodGuess-MLOps</h1>
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
