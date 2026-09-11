function BarIndicator({ label, value, tone }) {
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100
  return (
    <div className="bar-indicator">
      <div className="bar-indicator-head">
        <span className="bar-label">{label}</span>
        <span className="bar-value numeric">{pct.toFixed(1)}%</span>
      </div>
      <div className="bar-track">
        <div className={`bar-fill bar-fill-${tone}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function TrendStrip({ trend }) {
  if (!trend) {
    return (
      <section className="trend-strip">
        <p className="placeholder-text">Batch trend appears after the first inspection.</p>
      </section>
    )
  }

  return (
    <section className="trend-strip">
      <div className="trend-header">
        <span>
          Batch <span className="numeric">{trend.batch_id}</span>
        </span>
        <span className="numeric">n={trend.sample_size}</span>
      </div>

      <div className="trend-bars">
        <BarIndicator label="Defect rate" value={trend.defect_rate} tone="amber" />
        <BarIndicator label="Scrap rate" value={trend.scrap_rate} tone="red" />
      </div>

      {trend.drift_flag && <p className="drift-warning">Drift detected for this batch.</p>}

      {trend.note && <p className="disclaimer-text">{trend.note}</p>}
    </section>
  )
}

export default TrendStrip
