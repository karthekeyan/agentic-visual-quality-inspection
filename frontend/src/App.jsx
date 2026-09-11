import { useCallback, useEffect, useState } from 'react'
import UploadPane from './components/UploadPane'
import ResultPane from './components/ResultPane'
import TrendStrip from './components/TrendStrip'
import HistoryList from './components/HistoryList'
import { inspectImage, fetchHistory } from './api/client'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState('idle')
  const [report, setReport] = useState(null)
  const [errorMessage, setErrorMessage] = useState(null)
  const [history, setHistory] = useState([])
  const [historyError, setHistoryError] = useState(null)

  const loadHistory = useCallback(() => {
    fetchHistory(5)
      .then((data) => {
        setHistory(data.records)
        setHistoryError(null)
      })
      .catch(() => setHistoryError('History unavailable.'))
  }, [])

  useEffect(() => {
    loadHistory()
  }, [loadHistory])

  const handleFileSelected = (selectedFile) => {
    setFile(selectedFile)
    setReport(null)
    setErrorMessage(null)
    setStatus('idle')
    setPreviewUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous)
      return URL.createObjectURL(selectedFile)
    })
  }

  const handleSubmit = async () => {
    if (!file) return
    setStatus('analyzing')
    setErrorMessage(null)
    try {
      const data = await inspectImage(file)
      setReport(data)
      setStatus('done')
      loadHistory()
    } catch {
      setErrorMessage('Could not analyze this image. Check the file and try again.')
      setStatus('error')
    }
  }

  return (
    <div className="console">
      <header className="console-header">
        <h1>Visual QC Inspection Console</h1>
      </header>

      <main className="console-main">
        <UploadPane
          key={report?.generated_at ?? 'no-report'}
          previewUrl={previewUrl}
          overlayBase64={report?.overlay_image_base64}
          status={status}
          hasFile={!!file}
          onFileSelected={handleFileSelected}
          onSubmit={handleSubmit}
        />
        <ResultPane report={report} status={status} errorMessage={errorMessage} />
      </main>

      <TrendStrip trend={report?.trend} />
      <HistoryList records={history} error={historyError} />
    </div>
  )
}

export default App
