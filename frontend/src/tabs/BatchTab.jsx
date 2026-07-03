import { useEffect, useRef, useState } from 'react'
import {
  submitBatchPrediction,
  getBatchStatus,
  getBatchResults,
  cancelBatchJob,
  getBatchQueueStatus,
} from '../api'

export default function BatchTab() {
  const [lines, setLines] = useState(
    'Great product, will buy again!\nNever arrived, terrible service.\nAverage experience, nothing special.'
  )
  const [job, setJob] = useState(null)
  const [results, setResults] = useState(null)
  const [queue, setQueue] = useState(null)
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const pollRef = useRef(null)

  useEffect(() => {
    refreshQueue()
    return () => clearInterval(pollRef.current)
  }, [])

  function refreshQueue() {
    getBatchQueueStatus()
      .then(setQueue)
      .catch(() => setQueue(null))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    const texts = lines
      .split('\n')
      .map((t) => t.trim())
      .filter(Boolean)
    if (texts.length === 0) return

    setError(null)
    setResults(null)
    setSubmitting(true)

    try {
      const created = await submitBatchPrediction(texts)
      setJob(created)
      startPolling(created.job_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  function startPolling(jobId) {
    clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const status = await getBatchStatus(jobId)
        setJob(status)
        if (status.status === 'completed') {
          clearInterval(pollRef.current)
          const finalResults = await getBatchResults(jobId)
          setResults(finalResults)
          refreshQueue()
        } else if (status.status === 'failed') {
          clearInterval(pollRef.current)
          setError(status.error || 'Batch job failed')
        }
      } catch (err) {
        clearInterval(pollRef.current)
        setError(err.message)
      }
    }, 1000)
  }

  async function handleCancel() {
    if (!job) return
    try {
      await cancelBatchJob(job.job_id)
      clearInterval(pollRef.current)
      setJob({ ...job, status: 'cancelled' })
      refreshQueue()
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div>
      <p className="subtitle">
        Submits multiple texts to <code>/batch/predict</code>, polls{' '}
        <code>/batch/status/:id</code>, then fetches <code>/batch/results/:id</code> — real
        async job lifecycle, not a fake progress bar.
      </p>

      <form className="analyze-form" onSubmit={handleSubmit}>
        <textarea
          value={lines}
          onChange={(e) => setLines(e.target.value)}
          placeholder="One text per line..."
          rows={5}
        />
        <button type="submit" className="analyze-btn" disabled={submitting}>
          {submitting ? 'Submitting…' : 'Submit Batch'}
        </button>
      </form>

      {error && <div className="error-box">Error: {error}</div>}

      {job && (
        <div className="result-card">
          <div className="result-label" style={{ fontSize: 16 }}>
            Job {job.job_id.slice(0, 8)}… — {job.status}
          </div>
          <dl className="result-meta">
            <dt>Progress</dt>
            <dd>{((job.progress_percentage ?? 0) * 100).toFixed(0)}%</dd>
            <dt>Total texts</dt>
            <dd>{job.total_texts}</dd>
            {job.processed_texts !== undefined && (
              <>
                <dt>Processed</dt>
                <dd>{job.processed_texts}</dd>
              </>
            )}
          </dl>
          {(job.status === 'pending' || job.status === 'processing') && (
            <button type="button" className="feedback-btn" onClick={handleCancel}>
              Cancel job
            </button>
          )}
        </div>
      )}

      {results && (
        <div className="result-card">
          <div className="result-label" style={{ fontSize: 16 }}>
            Results ({results.total_results})
          </div>
          <ul className="batch-results">
            {results.results.map((r) => (
              <li key={r.prediction_id} className={`batch-result-${r.label.toLowerCase()}`}>
                <span className="batch-result-label">{r.label}</span>
                <span>{(r.score * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {queue && (
        <div className="queue-status">
          Queue — high: {queue.queues.high_priority} · medium: {queue.queues.medium_priority} ·
          low: {queue.queues.low_priority} · service: {queue.service_status}
        </div>
      )}
    </div>
  )
}
