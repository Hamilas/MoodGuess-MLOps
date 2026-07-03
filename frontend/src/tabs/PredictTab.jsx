import { useState } from 'react'
import { predictSentiment, submitFeedback } from '../api'

const EXAMPLES = [
  'I absolutely love this product, it works amazingly well!',
  'This is terrible, worst experience of my life.',
  'The delivery arrived on time and the packaging was solid.',
  'Customer support never responded to my emails.',
]

export default function PredictTab() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [feedbackSent, setFeedbackSent] = useState(false)

  async function handleAnalyze(e) {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed) return

    setLoading(true)
    setError(null)
    setResult(null)
    setFeedbackSent(false)

    try {
      const data = await predictSentiment(trimmed)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleFeedback(correctedLabel) {
    if (!result) return
    try {
      await submitFeedback(result.prediction_id, correctedLabel)
      setFeedbackSent(true)
    } catch (err) {
      setError(`Feedback failed: ${err.message}`)
    }
  }

  return (
    <div>
      <p className="subtitle">
        Live sentiment analysis — this form calls the real FastAPI backend at
        <code> /predict</code>, not a simulation.
      </p>

      <form className="analyze-form" onSubmit={handleAnalyze}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type a sentence to analyze..."
          rows={4}
        />

        <div className="examples">
          {EXAMPLES.map((example) => (
            <button
              type="button"
              key={example}
              className="chip"
              onClick={() => setText(example)}
            >
              {example.length > 40 ? `${example.slice(0, 40)}…` : example}
            </button>
          ))}
        </div>

        <button type="submit" className="analyze-btn" disabled={loading || !text.trim()}>
          {loading ? 'Analyzing…' : 'Analyze Sentiment'}
        </button>
      </form>

      {error && <div className="error-box">Error: {error}</div>}

      {result && (
        <div className={`result-card result-${result.label.toLowerCase()}`}>
          <div className="result-label">{result.label}</div>
          <div className="result-score">{(result.score * 100).toFixed(1)}% confidence</div>
          <dl className="result-meta">
            <dt>Backend</dt>
            <dd>{result.backend}</dd>
            <dt>Model</dt>
            <dd>{result.model_name}</dd>
            <dt>Latency</dt>
            <dd>{result.inference_time_ms.toFixed(2)} ms</dd>
            <dt>Cached</dt>
            <dd>{result.cached ? 'Yes' : 'No'}</dd>
          </dl>

          <div className="feedback-row">
            <span>Was this correct?</span>
            {feedbackSent ? (
              <span className="feedback-sent">Thanks for the feedback!</span>
            ) : (
              <>
                <button
                  type="button"
                  className="feedback-btn"
                  onClick={() => handleFeedback(result.label)}
                >
                  Yes
                </button>
                <button
                  type="button"
                  className="feedback-btn"
                  onClick={() =>
                    handleFeedback(result.label === 'POSITIVE' ? 'NEGATIVE' : 'POSITIVE')
                  }
                >
                  No, it's wrong
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
