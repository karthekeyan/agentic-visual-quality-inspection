import { useState } from 'react'
import UploadPane from './components/UploadPane'
import ResultPane from './components/ResultPane'
import { inspectImage } from './api/client'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState('idle')
  const [report, setReport] = useState(null)
  const [errorMessage, setErrorMessage] = useState(null)

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

  const handleReset = () => {
    setPreviewUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous)
      return null
    })
    setFile(null)
    setReport(null)
    setErrorMessage(null)
    setStatus('idle')
  }

  const handleSubmit = async () => {
    if (!file) return
    setStatus('analyzing')
    setErrorMessage(null)
    try {
      const data = await inspectImage(file)
      setReport(data)
      setStatus('done')
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
          boundingBox={report?.inspection?.bounding_box}
          status={status}
          hasFile={!!file}
          onFileSelected={handleFileSelected}
          onSubmit={handleSubmit}
          onReset={handleReset}
        />
        <ResultPane report={report} status={status} errorMessage={errorMessage} />
      </main>
    </div>
  )
}

export default App
