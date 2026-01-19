import { useState, useCallback } from 'react'
import { uploadProfile, type Session } from '../api'

interface Props {
  session: Session
  onProfileLoaded: () => void
}

type Status = 'idle' | 'dragging' | 'uploading' | 'done' | 'error'

export function ProfileUpload({ session, onProfileLoaded }: Props) {
  const [status, setStatus] = useState<Status>('idle')
  const [fileName, setFileName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith('.md') && !file.name.endsWith('.txt')) {
        setError('Please upload a .md or .txt file')
        setStatus('error')
        return
      }

      setFileName(file.name)
      setStatus('uploading')
      setError(null)

      try {
        const content = await file.text()
        await uploadProfile(session.id, content)
        setStatus('done')
        onProfileLoaded()
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Upload failed')
        setStatus('error')
      }
    },
    [session.id, onProfileLoaded]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setStatus('idle')
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setStatus('dragging')
  }

  const handleDragLeave = () => {
    setStatus('idle')
  }

  const handleClick = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.md,.txt'
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (file) handleFile(file)
    }
    input.click()
  }

  return (
    <div
      style={{
        ...styles.dropzone,
        ...(status === 'dragging' ? styles.dragging : {}),
        ...(status === 'done' ? styles.done : {}),
        ...(status === 'error' ? styles.error : {}),
      }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={status !== 'done' ? handleClick : undefined}
    >
      {status === 'idle' && (
        <>
          <div style={styles.icon}>📄</div>
          <div style={styles.text}>Drop your profile here</div>
          <div style={styles.hint}>or click to browse (.md, .txt)</div>
        </>
      )}

      {status === 'dragging' && (
        <>
          <div style={styles.icon}>📥</div>
          <div style={styles.text}>Drop to upload</div>
        </>
      )}

      {status === 'uploading' && (
        <>
          <div style={styles.spinner} />
          <div style={styles.text}>Processing {fileName}...</div>
        </>
      )}

      {status === 'done' && (
        <>
          <div style={styles.icon}>✓</div>
          <div style={styles.text}>{fileName}</div>
          <div style={styles.hint}>Profile loaded</div>
        </>
      )}

      {status === 'error' && (
        <>
          <div style={styles.icon}>⚠</div>
          <div style={styles.text}>{error}</div>
          <div style={styles.hint}>Click to try again</div>
        </>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  dropzone: {
    border: '2px dashed #333',
    borderRadius: '12px',
    padding: '3rem 2rem',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.2s',
    background: '#111',
  },
  dragging: {
    borderColor: '#3b82f6',
    background: '#1e3a5f20',
  },
  done: {
    borderColor: '#22c55e',
    borderStyle: 'solid',
    background: '#14532d20',
    cursor: 'default',
  },
  error: {
    borderColor: '#ef4444',
    background: '#7f1d1d20',
  },
  icon: {
    fontSize: '2.5rem',
    marginBottom: '1rem',
  },
  text: {
    fontSize: '1.1rem',
    color: '#e5e5e5',
  },
  hint: {
    fontSize: '0.875rem',
    color: '#666',
    marginTop: '0.5rem',
  },
  spinner: {
    width: '32px',
    height: '32px',
    border: '3px solid #333',
    borderTopColor: '#3b82f6',
    borderRadius: '50%',
    margin: '0 auto 1rem',
    animation: 'spin 1s linear infinite',
  },
}

// Add keyframes via style tag
if (typeof document !== 'undefined' && !document.getElementById('spinner-style')) {
  const style = document.createElement('style')
  style.id = 'spinner-style'
  style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }'
  document.head.appendChild(style)
}
