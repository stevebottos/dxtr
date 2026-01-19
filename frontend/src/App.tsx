import { useState, useEffect } from 'react'
import { Chat } from './components/Chat'
import { createSession, clearSession, type Session } from './api'

export default function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [resetKey, setResetKey] = useState(0)

  useEffect(() => {
    createSession()
      .then(setSession)
      .catch((e) => setError(e.message))
  }, [resetKey])

  const handleClear = () => {
    if (window.confirm('Are you sure you want to clear the chat history?')) {
      clearSession()
      setSession(null)
      setResetKey(prev => prev + 1)
    }
  }

  if (error) {
    return (
      <div style={styles.container}>
        <div style={styles.error}>
          <h2>Error</h2>
          <p>{error}</p>
          <button onClick={() => window.location.reload()} style={styles.retryButton}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!session) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Initializing...</div>
      </div>
    )
  }

  return (
    <>
      <div style={styles.controls}>
        <button onClick={handleClear} style={styles.clearButton} title="Clear Chat History">
          🗑️
        </button>
      </div>
      <Chat key={session.id} session={session} />
    </>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem',
    background: '#111',
    color: '#eee',
  },
  loading: {
    color: '#888',
  },
  error: {
    color: '#f87171',
    textAlign: 'center',
  },
  retryButton: {
    marginTop: '1rem',
    padding: '0.5rem 1rem',
    background: '#333',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  controls: {
    position: 'fixed',
    top: '1rem',
    right: '1rem',
    zIndex: 100,
  },
  clearButton: {
    background: '#222',
    border: '1px solid #333',
    color: '#fff',
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.2rem',
    transition: 'all 0.2s',
  },
}
