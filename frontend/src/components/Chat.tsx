import { useState, useRef, useEffect } from 'react'
import { streamChat, type Session, type StreamEvent } from '../api'

interface Message {
  role: 'user' | 'assistant'
  content: string
  logs?: StreamEvent[] // Attached logs for this turn
}

interface Props {
  session: Session
}

export function Chat({ session }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  
  // Temporary logs for the current streaming response
  const [currentLogs, setCurrentLogs] = useState<StreamEvent[]>([])
  
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentLogs])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isStreaming) return

    const userMessage = input.trim()
    setInput('')
    
    // Add user message immediately
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    
    setIsStreaming(true)
    setCurrentLogs([])

    try {
      let finalAnswer = ''

      for await (const event of streamChat(session.id, userMessage)) {
        if (event.type === 'tool' || event.type === 'status') {
          setCurrentLogs((prev) => [...prev, event])
        } else if (event.type === 'done') {
          finalAnswer = event.answer || ''
        } else if (event.type === 'error') {
          setCurrentLogs((prev) => [...prev, { type: 'error', message: event.message }])
          finalAnswer = `Error: ${event.message}`
        }
      }

      // Add the final assistant message with the accumulated logs
      setMessages((prev) => [
        ...prev,
        { 
          role: 'assistant', 
          content: finalAnswer,
          logs: [...currentLogs] // Capture the logs that happened during this turn
        },
      ])

    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${err instanceof Error ? err.message : 'Unknown error'}`,
        },
      ])
    } finally {
      setIsStreaming(false)
      setCurrentLogs([]) // Clear live logs as they are now attached to the message
    }
  }

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.logo}>DXTR</h1>
        <div style={styles.status}>
            <span style={styles.badge}>{session.model}</span>
        </div>
      </header>

      <div style={styles.messages}>
        {messages.length === 0 && (
          <div style={styles.empty}>
            <p>Start a conversation.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              ...styles.messageWrapper,
              alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
            }}
          >
            {/* Render attached logs for assistant messages */}
            {msg.role === 'assistant' && msg.logs && msg.logs.length > 0 && (
              <div style={styles.logsContainer}>
                {msg.logs.map((log, idx) => (
                  <div key={idx} style={styles.logItem}>
                     <span style={styles.logIcon}>
                       {log.type === 'tool' ? '⚙️' : 'ℹ️'}
                     </span>
                     {log.message}
                  </div>
                ))}
              </div>
            )}

            <div
              style={{
                ...styles.message,
                ...(msg.role === 'user' ? styles.userMessage : styles.assistantMessage),
              }}
            >
              <div style={styles.messageRole}>
                {msg.role === 'user' ? 'You' : 'DXTR'}
              </div>
              <div style={styles.messageContent}>{msg.content}</div>
            </div>
          </div>
        ))}

        {/* Live Streaming Area */}
        {isStreaming && (
            <div style={{...styles.messageWrapper, alignItems: 'flex-start'}}>
                
                {/* Live Logs */}
                {currentLogs.length > 0 && (
                  <div style={styles.logsContainer}>
                    {currentLogs.map((log, idx) => (
                      <div key={idx} style={{...styles.logItem, opacity: 0.8}}>
                        <span style={styles.logIcon}>
                          {log.type === 'tool' ? '⚙️' : 'ℹ️'}
                        </span>
                        {log.message}
                      </div>
                    ))}
                    <div style={styles.thinking}>Running...</div>
                  </div>
                )}
                
                {/* Placeholder while thinking */}
                {currentLogs.length === 0 && (
                    <div style={{...styles.message, ...styles.assistantMessage}}>
                        <div style={styles.messageRole}>DXTR</div>
                        <div style={styles.messageContent}>Thinking...</div>
                    </div>
                )}
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form style={styles.inputArea} onSubmit={handleSubmit}>
        <input
          style={styles.input}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isStreaming}
        />
        <button
          type="submit"
          style={{
            ...styles.sendButton,
            opacity: isStreaming || !input.trim() ? 0.5 : 1,
          }}
          disabled={isStreaming || !input.trim()}
        >
          Send
        </button>
      </form>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    maxWidth: '800px',
    margin: '0 auto',
    padding: '0 1rem',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    background: '#111',
    color: '#eee',
  },
  header: {
    padding: '1rem 0',
    borderBottom: '1px solid #333',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  logo: {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    margin: 0,
    background: 'linear-gradient(135deg, #60a5fa, #a78bfa)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  status: {
    display: 'flex',
    gap: '1rem',
    alignItems: 'center',
  },
  badge: {
    background: '#222',
    padding: '0.25rem 0.5rem',
    borderRadius: '4px',
    fontSize: '0.8rem',
    color: '#888',
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '1rem 0',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  empty: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#555',
  },
  messageWrapper: {
    display: 'flex',
    flexDirection: 'column',
    width: '100%',
  },
  message: {
    padding: '1rem',
    borderRadius: '8px',
    maxWidth: '80%',
    position: 'relative',
  },
  userMessage: {
    background: '#1e3a5f',
    borderBottomRightRadius: '2px',
  },
  assistantMessage: {
    background: '#222',
    borderBottomLeftRadius: '2px',
  },
  messageRole: {
    fontSize: '0.7rem',
    color: '#aaa',
    marginBottom: '0.25rem',
    textTransform: 'uppercase',
  },
  messageContent: {
    lineHeight: '1.5',
    whiteSpace: 'pre-wrap',
  },
  logsContainer: {
    marginBottom: '0.5rem',
    padding: '0.5rem',
    background: '#151515',
    border: '1px solid #222',
    borderRadius: '6px',
    fontSize: '0.8rem',
    color: '#888',
    maxWidth: '80%',
    alignSelf: 'flex-start',
  },
  logItem: {
    display: 'flex',
    gap: '0.5rem',
    marginBottom: '0.25rem',
    alignItems: 'center',
  },
  logIcon: {
    fontSize: '0.9rem',
  },
  thinking: {
    fontStyle: 'italic',
    color: '#555',
    marginTop: '0.25rem',
    marginLeft: '1.4rem',
  },
  inputArea: {
    padding: '1rem 0',
    borderTop: '1px solid #333',
    display: 'flex',
    gap: '0.5rem',
  },
  input: {
    flex: 1,
    padding: '0.75rem',
    background: '#222',
    border: '1px solid #333',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '1rem',
  },
  sendButton: {
    padding: '0 1.5rem',
    background: '#3b82f6',
    border: 'none',
    borderRadius: '4px',
    color: 'white',
    fontSize: '1rem',
    fontWeight: 'bold',
    cursor: 'pointer',
  },
}
