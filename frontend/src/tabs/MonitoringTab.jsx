import { useEffect, useState } from 'react'
import {
  getBusinessKpis,
  getQualityKpis,
  getCostKpis,
  getPerformanceKpis,
  getSloStatus,
  getModelInfo,
  getDriftSummary,
  checkDrift,
  getReady,
} from '../api'

function StatCard({ title, children }) {
  return (
    <div className="stat-card">
      <div className="stat-title">{title}</div>
      {children}
    </div>
  )
}

export default function MonitoringTab() {
  const [business, setBusiness] = useState(null)
  const [quality, setQuality] = useState(null)
  const [cost, setCost] = useState(null)
  const [performance, setPerformance] = useState(null)
  const [slo, setSlo] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [drift, setDrift] = useState(null)
  const [ready, setReady] = useState(null)
  const [error, setError] = useState(null)

  function loadAll() {
    setError(null)
    Promise.allSettled([
      getBusinessKpis(),
      getQualityKpis(),
      getCostKpis(),
      getPerformanceKpis(),
      getSloStatus(),
      getModelInfo(),
      getDriftSummary(),
      getReady(),
    ]).then(
      ([
        businessRes,
        qualityRes,
        costRes,
        performanceRes,
        sloRes,
        modelRes,
        driftRes,
        readyRes,
      ]) => {
        if (businessRes.status === 'fulfilled') setBusiness(businessRes.value)
        if (qualityRes.status === 'fulfilled') setQuality(qualityRes.value)
        if (costRes.status === 'fulfilled') setCost(costRes.value)
        if (performanceRes.status === 'fulfilled') setPerformance(performanceRes.value)
        if (sloRes.status === 'fulfilled') setSlo(sloRes.value)
        if (modelRes.status === 'fulfilled') setModelInfo(modelRes.value)
        if (driftRes.status === 'fulfilled') setDrift(driftRes.value)
        if (readyRes.status === 'fulfilled') setReady(readyRes.value)
      }
    )
  }

  useEffect(() => {
    loadAll()
  }, [])

  async function handleCheckDrift() {
    try {
      const result = await checkDrift()
      setDrift(result)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div>
      <p className="subtitle">
        Live values from <code>/monitoring/kpis/*</code>, <code>/model-info</code>,{' '}
        <code>/monitoring/drift</code>, and <code>/ready</code> — refresh to re-fetch real
        numbers from the running service.
      </p>

      <button type="button" className="analyze-btn" style={{ marginBottom: 16 }} onClick={loadAll}>
        Refresh
      </button>

      {error && <div className="error-box">Error: {error}</div>}

      <div className="stat-grid">
        <StatCard title="Business">
          {business ? (
            <>
              <div className="stat-value">{business.total_predictions}</div>
              <div className="stat-sub">
                {(business.positive_ratio * 100).toFixed(0)}% positive ·{' '}
                {(business.negative_ratio * 100).toFixed(0)}% negative
              </div>
            </>
          ) : (
            <div className="stat-sub">unavailable</div>
          )}
        </StatCard>

        <StatCard title="Quality">
          {quality ? (
            <>
              <div className="stat-value">{(quality.confidence_p50 * 100).toFixed(0)}%</div>
              <div className="stat-sub">median confidence · p95 {(quality.confidence_p95 * 100).toFixed(0)}%</div>
            </>
          ) : (
            <div className="stat-sub">unavailable</div>
          )}
        </StatCard>

        <StatCard title="Cost">
          {cost ? (
            <>
              <div className="stat-value">${cost.total_cost_usd.toFixed(5)}</div>
              <div className="stat-sub">{cost.cache_savings_percent.toFixed(0)}% saved by cache</div>
            </>
          ) : (
            <div className="stat-sub">unavailable</div>
          )}
        </StatCard>

        <StatCard title="Performance">
          {performance ? (
            <>
              <div className="stat-value">{performance.latency_p95_ms.toFixed(1)} ms</div>
              <div className="stat-sub">p95 latency · {performance.throughput_rps.toFixed(2)} req/s</div>
            </>
          ) : (
            <div className="stat-sub">unavailable</div>
          )}
        </StatCard>

        <StatCard title="SLO">
          {slo ? (
            <>
              <div className={`stat-value ${slo.compliant ? 'stat-good' : 'stat-bad'}`}>
                {slo.compliant ? 'Compliant' : 'Breached'}
              </div>
              <div className="stat-sub">availability {slo.availability.current.toFixed(1)}%</div>
            </>
          ) : (
            <div className="stat-sub">unavailable</div>
          )}
        </StatCard>

        <StatCard title="Readiness">
          {ready ? (
            <>
              <div className="stat-value stat-good">Ready</div>
              <div className="stat-sub">k8s readiness probe passing</div>
            </>
          ) : (
            <div className="stat-sub">unavailable</div>
          )}
        </StatCard>
      </div>

      <div className="result-card">
        <div className="result-label" style={{ fontSize: 16 }}>
          Model Info
        </div>
        {modelInfo ? (
          <dl className="result-meta">
            <dt>Name</dt>
            <dd>{modelInfo.model_name}</dd>
            <dt>Backend</dt>
            <dd>{modelInfo.backend}</dd>
            <dt>Version</dt>
            <dd>{modelInfo.version}</dd>
            <dt>Labels</dt>
            <dd>{modelInfo.labels?.join(', ')}</dd>
          </dl>
        ) : (
          <div className="stat-sub">unavailable</div>
        )}
      </div>

      <div className="result-card">
        <div className="result-label" style={{ fontSize: 16 }}>
          Drift Detection
        </div>
        {drift?.enabled === false ? (
          <div className="stat-sub">{drift.message}</div>
        ) : drift ? (
          <dl className="result-meta">
            <dt>Data drift</dt>
            <dd>{String(drift.data_drift_detected)}</dd>
            <dt>Prediction drift</dt>
            <dd>{String(drift.prediction_drift_detected)}</dd>
            <dt>Drift score</dt>
            <dd>{drift.drift_score}</dd>
          </dl>
        ) : (
          <div className="stat-sub">unavailable</div>
        )}
        <button type="button" className="feedback-btn" onClick={handleCheckDrift}>
          Check drift now
        </button>
      </div>
    </div>
  )
}
