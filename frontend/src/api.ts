export interface Session {
  id: string;
  provider: string;
  model: string;
}

export type EventType = 'tool' | 'status' | 'done' | 'error';

export interface StreamEvent {
  type: EventType;
  message?: string;
  answer?: string;
}

const SESSION_KEY = 'dxtr_session_id';

export async function createSession(): Promise<Session> {
  let sessionId = localStorage.getItem(SESSION_KEY);
  
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
  }

  return Promise.resolve({
    id: sessionId,
    provider: 'local',
    model: 'dxtr-agent',
  });
}

export function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

export async function getHistory(sessionId: string): Promise<ChatMessage[]> {
  const res = await fetch(`/api/history/${sessionId}?user_id=web-user`);
  if (!res.ok) {
    console.warn('Failed to fetch history');
    return [];
  }
  return res.json();
}

export async function* streamChat(
  sessionId: string,
  message: string
): AsyncGenerator<StreamEvent> {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: 'web-user',
      session_id: sessionId,
      query: message,
    }),
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Chat failed: ${res.status} ${errorText}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const dataStr = line.slice(6);
        if (!dataStr) continue;
        
        try {
          const event = JSON.parse(dataStr) as StreamEvent;
          yield event;
        } catch (e) {
          console.error('Failed to parse SSE data:', dataStr, e);
        }
      }
    }
  }
}