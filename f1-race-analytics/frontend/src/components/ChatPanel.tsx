import { useRef, useState } from "react"

import { PitWallAvatar } from "@/components/PitWallAvatar"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { type ChatMessage, sendChatMessage } from "@/lib/api"

// The backend is stateless (see backend/api/main.py) — we hold the full
// conversation client-side and resend it every request. `chat()` on the
// backend returns the ENTIRE conversation back (original messages + new
// assistant/tool turns appended), so we can just replace state with the
// response rather than manually merging.
export function ChatPanel({ selectedSessionKey, selectedRaceLabel }: { selectedSessionKey: number | null; selectedRaceLabel: string | null }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [pending, setPending] = useState(false)
  const lastContextSessionKey = useRef<number | null>(null)

  async function handleSend() {
    const text = input.trim()
    if (!text || pending) return

    const outgoing = [...messages]
    // Tell the model which race is on screen, once per selection change —
    // not shown in the chat log itself (filtered out below by its
    // "[Context:" prefix), just context riding along with the next message.
    if (selectedSessionKey && lastContextSessionKey.current !== selectedSessionKey) {
      outgoing.push({ role: "user", content: `[Context: user is currently viewing ${selectedRaceLabel}, session_key=${selectedSessionKey}]` })
      lastContextSessionKey.current = selectedSessionKey
    }
    outgoing.push({ role: "user", content: text })

    setMessages(outgoing)
    setInput("")
    setPending(true)
    try {
      const result = await sendChatMessage(outgoing)
      setMessages(result)
    } catch {
      setMessages([...outgoing, { role: "assistant", content: "Something went wrong talking to the backend — is it running?" }])
    } finally {
      setPending(false)
    }
  }

  const visible = messages.filter(
    (m) => (m.role === "user" || m.role === "assistant") && m.content && !m.content.startsWith("[Context:")
  )

  return (
    <div className="flex h-[600px] flex-col gap-3">
      <ScrollArea className="border-border bg-card/40 flex-1 rounded-lg border p-4">
        <div className="flex flex-col gap-3">
          {visible.length === 0 && (
            <p className="text-muted-foreground text-sm">
              Ask about a race — tyre strategy, pit stops, who gained positions. Pick a race above first so I know what you're talking about.
            </p>
          )}
          {visible.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="self-end">
                <div className="bg-primary text-primary-foreground max-w-md rounded-lg rounded-br-sm px-3 py-2 text-sm">{m.content}</div>
              </div>
            ) : (
              <div key={i} className="flex max-w-md items-start gap-2 self-start">
                <PitWallAvatar size={28} />
                <div className="bg-secondary text-secondary-foreground rounded-lg rounded-bl-sm px-3 py-2 text-sm whitespace-pre-line">{m.content}</div>
              </div>
            )
          )}
          {pending && (
            <div className="flex items-center gap-2 self-start">
              <PitWallAvatar size={28} />
              <div className="text-muted-foreground text-sm">Pit Wall is thinking…</div>
            </div>
          )}
        </div>
      </ScrollArea>
      <div className="flex gap-2">
        <input
          className="border-border bg-card focus-visible:ring-ring flex-1 rounded-lg border px-3 py-2 text-sm outline-none focus-visible:ring-2"
          placeholder="What happened at turn 1?"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={pending}
        />
        <Button onClick={handleSend} disabled={pending || !input.trim()}>
          Send
        </Button>
      </div>
    </div>
  )
}
