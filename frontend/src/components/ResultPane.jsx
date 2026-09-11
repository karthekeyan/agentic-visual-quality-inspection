import { statusColor } from './statusColor'
import { renderFormattedText } from './formatText'

function ResultPane({ report, status, errorMessage }) {
  if (status === 'error') {
    return (
      <section className="pane pane-result">
        <p className="error-text">{errorMessage}</p>
      </section>
    )
  }

  if (status === 'analyzing') {
    return (
      <section className="pane pane-result">
        <p className="status-text">Analyzing…</p>
      </section>
    )
  }

  if (!report) {
    return (
      <section className="pane pane-result">
        <p className="placeholder-text">Upload an image and submit for inspection to see results here.</p>
      </section>
    )
  }

  const { inspection, characterization, root_cause: rootCause, disposition, human_review_required: humanReviewRequired } = report
  const labelColor = statusColor(inspection.label)
  const dispositionColor = statusColor(disposition.decision)

  return (
    <section className="pane pane-result">
      {humanReviewRequired && <div className="banner-review">Human review required</div>}

      <div className={`readout-block accent-${labelColor}`}>
        <div className="block-label">Label</div>
        <div className="block-main">
          <span className="label-text">{inspection.label}</span>
          <span className="confidence-text numeric">{(inspection.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      {characterization && (
        <div className={`readout-block accent-${dispositionColor}`}>
          <div className="block-label">Characterization</div>
          <div className="block-body">
            <p>
              <strong>{characterization.category}</strong> — {characterization.description}
            </p>
            <dl className="char-meta">
              <div>
                <dt>Region size</dt>
                <dd className="numeric">{characterization.region_size}</dd>
              </div>
              <div>
                <dt>Position</dt>
                <dd>{characterization.position}</dd>
              </div>
              <div>
                <dt>Confidence tier</dt>
                <dd>{characterization.confidence_tier}</dd>
              </div>
            </dl>
            {characterization.heuristic_approximation && (
              <p className="muted-note">Heuristic approximation.</p>
            )}
          </div>
        </div>
      )}

      {rootCause && (
        <div className={`readout-block accent-${dispositionColor}`}>
          <div className="block-label">Root cause</div>
          <div className="block-body root-cause-text">{renderFormattedText(rootCause.explanation)}</div>
        </div>
      )}

      <div className={`readout-block accent-${dispositionColor}`}>
        <div className="block-label">Disposition</div>
        <div className="block-body">
          <p className="disposition-decision">{disposition.decision}</p>
          <p className="disposition-reasoning">{disposition.reasoning}</p>
        </div>
      </div>
    </section>
  )
}

export default ResultPane
