/**
 * Maps pipeline status words (label / disposition decision) to the design
 * system's three accent tones. Red is reserved for 'scrap' only.
 */
export function statusColor(status) {
  const value = (status || '').toLowerCase()
  if (value === 'scrap') return 'red'
  if (value === 'ok' || value === 'accept') return 'teal'
  return 'amber'
}
