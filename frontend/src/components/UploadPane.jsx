import { useRef, useState } from 'react'

function UploadPane({ previewUrl, overlayBase64, boundingBox, status, onFileSelected, onSubmit, onReset, hasFile }) {
  const [isDragging, setIsDragging] = useState(false)
  // null until the viewer picks explicitly; defaults to side-by-side once an
  // overlay exists. Remounted per-report (see App.jsx `key`) so this resets
  // automatically for each new inspection without syncing via an effect.
  const [manualView, setManualView] = useState(null)
  const view = manualView ?? (overlayBase64 ? 'side-by-side' : 'original')
  const inputRef = useRef(null)
  // Bounding-box coordinates come back in the overlay image's own pixel
  // space (see src.inference.gradcam.compute_bounding_box), so drawing the
  // marker as a percentage needs the displayed overlay <img>'s natural
  // (unscaled) size, only known once it has actually loaded.
  const [naturalSize, setNaturalSize] = useState(null)

  const handleFiles = (files) => {
    const selected = files && files[0]
    if (selected && selected.type.startsWith('image/')) {
      onFileSelected(selected)
    }
  }

  const showOriginal = view === 'original' || view === 'side-by-side'
  const showOverlay = !!overlayBase64 && (view === 'overlay' || view === 'side-by-side')
  // The box always draws on the overlay specifically, in any view that
  // shows the overlay -- never on the plain original.
  const showBox = showOverlay && !!boundingBox && !!naturalSize

  return (
    <section className="pane pane-upload">
      <div
        className={`dropzone${isDragging ? ' dropzone-active' : ''}${previewUrl ? ' dropzone-has-image' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setIsDragging(false)
          handleFiles(event.dataTransfer.files)
        }}
      >
        {!previewUrl && (
          <div className="dropzone-prompt">
            <p>Drag and drop an image, or click to browse</p>
          </div>
        )}

        {previewUrl && (
          <div className="image-stage">
            {showOriginal && (
              <figure>
                <img src={previewUrl} alt="Uploaded part" />
                <figcaption>Original</figcaption>
              </figure>
            )}
            {showOverlay && (
              <figure>
                <div className="image-frame">
                  <img
                    src={`data:image/png;base64,${overlayBase64}`}
                    alt="Grad-CAM overlay"
                    onLoad={(e) => setNaturalSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
                  />
                  {showBox && (
                    <div
                      className="bbox-marker"
                      style={{
                        left: `${(boundingBox[0] / naturalSize.w) * 100}%`,
                        top: `${(boundingBox[1] / naturalSize.h) * 100}%`,
                        width: `${(boundingBox[2] / naturalSize.w) * 100}%`,
                        height: `${(boundingBox[3] / naturalSize.h) * 100}%`,
                      }}
                    />
                  )}
                </div>
                <figcaption>Grad-CAM overlay{showBox ? ' (defect boxed)' : ''}</figcaption>
              </figure>
            )}
          </div>
        )}

        {status === 'done' && (
          <button
            type="button"
            className="new-inspection-btn"
            onClick={(e) => {
              e.stopPropagation()
              onReset()
            }}
          >
            ← New inspection
          </button>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(event) => handleFiles(event.target.files)}
        />
      </div>

      <div className="upload-controls">
        {overlayBase64 && (
          <div className="view-toggle">
            <button
              type="button"
              className={view === 'original' ? 'active' : ''}
              onClick={(e) => {
                e.stopPropagation()
                setManualView('original')
              }}
            >
              Original
            </button>
            <button
              type="button"
              className={view === 'overlay' ? 'active' : ''}
              onClick={(e) => {
                e.stopPropagation()
                setManualView('overlay')
              }}
            >
              Grad-CAM
            </button>
            <button
              type="button"
              className={view === 'side-by-side' ? 'active' : ''}
              onClick={(e) => {
                e.stopPropagation()
                setManualView('side-by-side')
              }}
            >
              Side by side
            </button>
          </div>
        )}

        {status !== 'done' && (
          <button
            type="button"
            className="submit-btn"
            disabled={!hasFile || status === 'analyzing'}
            onClick={onSubmit}
          >
            {status === 'analyzing' ? 'Analyzing…' : 'Submit for inspection'}
          </button>
        )}
      </div>
    </section>
  )
}

export default UploadPane
