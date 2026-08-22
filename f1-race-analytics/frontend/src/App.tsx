import { useState } from "react"

import { ChartsView } from "@/components/ChartsView"
import { ChatPanel } from "@/components/ChatPanel"
import { PredictionView } from "@/components/PredictionView"
import { RaceSelector } from "@/components/RaceSelector"
import { SummaryView } from "@/components/SummaryView"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

function App() {
  const [selected, setSelected] = useState<{ sessionKey: number; label: string } | null>(null)

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-6 py-8">
      <header className="border-border flex flex-wrap items-center justify-between gap-4 border-b pb-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Pit <span className="text-primary">Wall</span>
          </h1>
          <p className="text-muted-foreground text-sm">F1 race analytics, grounded in real session data</p>
        </div>
        <RaceSelector selected={selected?.sessionKey ?? null} onSelect={(sessionKey, label) => setSelected({ sessionKey, label })} />
      </header>

      {!selected ? (
        <div className="text-muted-foreground flex flex-1 items-center justify-center rounded-lg border border-dashed py-24 text-sm">
          Pick a race above to see charts, a summary, and chat about it.
        </div>
      ) : (
        <Tabs defaultValue="charts">
          <TabsList>
            <TabsTrigger value="charts">Charts</TabsTrigger>
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="prediction">Prediction</TabsTrigger>
            <TabsTrigger value="chat">Chat</TabsTrigger>
          </TabsList>
          <TabsContent value="charts" className="mt-4">
            <ChartsView sessionKey={selected.sessionKey} />
          </TabsContent>
          <TabsContent value="summary" className="mt-4">
            <SummaryView sessionKey={selected.sessionKey} />
          </TabsContent>
          <TabsContent value="prediction" className="mt-4">
            <PredictionView sessionKey={selected.sessionKey} />
          </TabsContent>
          <TabsContent value="chat" className="mt-4">
            <ChatPanel selectedSessionKey={selected.sessionKey} selectedRaceLabel={selected.label} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  )
}

export default App
