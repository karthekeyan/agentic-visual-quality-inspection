function filenameOf(path) {
  if (!path) return '—'
  return path.split('/').pop()
}

function HistoryList({ records, error }) {
  return (
    <section className="history-list">
      <h2 className="history-heading">Recent inspections</h2>

      {error && <p className="muted-note">{error}</p>}
      {!error && records.length === 0 && <p className="muted-note">No inspection history yet.</p>}

      {!error && records.length > 0 && (
        <ul>
          {records.map((record, index) => (
            <li key={`${record.image_path ?? 'record'}-${index}`}>
              <span className="history-filename">{filenameOf(record.image_path)}</span>
              <span className="history-label">{record.label ?? '—'}</span>
              <span className="history-disposition">{record.disposition_decision ?? '—'}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export default HistoryList
