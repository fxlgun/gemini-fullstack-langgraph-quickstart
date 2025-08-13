// src/App.tsx  (updated)
// --- keep your existing imports, plus these ---
import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { auth, signOut } from "./firebase";
import { Button } from "@/components/ui/button";

export default function App() {
  // --- existing state and refs ---
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);
  const [error, setError] = useState<string | null>(null);

  // NEW: track the phone-based user identity and the thread_id we will use
  const [userPhone, setUserPhone] = useState<string | null>(null); // e.g. "+919876543210"
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [initializingThread, setInitializingThread] = useState(false);
  const [idToken, setIdToken] = useState<string | null>(null);

  // useStream as before
  const thread = useStream<{
    messages: Message[];
    reasoning_model: string;
    rag_query: string | null;
    rag_docs: string | null;
  }>({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "http://localhost:8123",
    assistantId: "agent",
    messagesKey: "messages",
    defaultHeaders: {
      Authorization: `Bearer ${idToken}`,
    },
    onUpdateEvent: (event: any) => {
      // unchanged event mapping
      let processedEvent: ProcessedEvent | null = null;
      console.log("Received event:", event);
      if (event.retrieve_rag_docs) {
        processedEvent = {
          title: "Retrieving data from Documents",
          data: `For information on ${event.retrieve_rag_docs.rag_query}`,
        };
      } else if (event.web_research) {
        const sources = event.web_research.sources_gathered || [];
        const numSources = sources.length;
        const uniqueLabels = [
          ...new Set(sources.map((s: any) => s.label).filter(Boolean)),
        ];
        const exampleLabels = uniqueLabels.slice(0, 3).join(", ");
        processedEvent = {
          title: "Web Research",
          data: `Gathered ${numSources} sources. Related to: ${
            exampleLabels || "N/A"
          }.`,
        };
      } else if (event.reflection) {
        processedEvent = {
          title: "Reflection",
          data: "Analysing Web Research Results",
        };
      } else if (event.finalize_answer) {
        processedEvent = {
          title: "Finalizing Answer",
          data: "Composing and presenting the final answer.",
        };
        hasFinalizeEventOccurredRef.current = true;
      } else if (event.answer_with_rag) {
        processedEvent = {
          title: "Finalizing Answer",
          data: `Composing and bringing you the final answer.`,
        };
      }
      if (processedEvent) {
        setProcessedEventsTimeline((prevEvents) => [
          ...prevEvents,
          processedEvent!,
        ]);
      }
    },
    onError: (error: any) => {
      setError(error.message);
    },
  });

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(async (user) => {
      if (user) {
        const token = await user.getIdToken();
        setIdToken(token);
        setUserPhone(user.phoneNumber || null); // store phone number
      } else {
        setIdToken(null);
        setUserPhone(null);
      }
    });

    return () => unsubscribe();
  }, []);

  // --- scroll behaviour + finalize saving (unchanged) ---
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [thread.messages]);

  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !thread.isLoading &&
      thread.messages.length > 0
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  // ---------------------------
  // Logout handler (unchanged)
  // ---------------------------
  const handleLogout = async () => {
    try {
      await signOut(auth);
      window.location.reload();
    } catch (err) {
      console.error("Logout error:", err);
    }
  };

  // ---------------------------
  // handleSubmit: uses the currentThreadId (if available)
  // ---------------------------
  const handleSubmit = useCallback(
    (submittedInputValue: string, model: string) => {
      if (!submittedInputValue.trim()) return;
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;

      // Build messages
      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];

      // If we have a thread_id, pass it through config so the run attaches to persisted thread
      if (currentThreadId) {
        thread.submit({
          messages: newMessages,
          reasoning_model: model,
          rag_query: null,
          rag_docs: null,
          // NOTE: some SDKs accept a separate options arg; here we embed a common pattern
          config: {
            configurable: { thread_id: currentThreadId } as any,
          } as any,
        } as any);
      } else {
        // No pre-created thread — fallback to prior behaviour (SDK auto-creates thread)
        thread.submit({
          messages: newMessages,
          reasoning_model: model,
          rag_query: null,
          rag_docs: null,
        });
      }
    },
    [thread, currentThreadId]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
    window.location.reload();
  }, [thread]);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      {/* place logout in top-right */}
      <header className="absolute top-4 right-4 z-50">
        <Button
          variant="secondary"
          size="sm"
          className="shadow-md"
          onClick={handleLogout}
        >
          Logout
        </Button>
      </header>

      <main className="h-full w-full max-w-4xl mx-auto">
        {thread.messages.length === 0 ? (
          <WelcomeScreen
            handleSubmit={handleSubmit}
            isLoading={thread.isLoading || initializingThread}
            onCancel={handleCancel}
          />
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="flex flex-col items-center justify-center gap-4">
              <h1 className="text-2xl text-red-400 font-bold">Error</h1>
              <p className="text-red-400">{JSON.stringify(error)}</p>

              <Button
                variant="destructive"
                onClick={() => window.location.reload()}
              >
                Retry
              </Button>
            </div>
          </div>
        ) : (
          <ChatMessagesView
            messages={thread.messages}
            isLoading={thread.isLoading}
            scrollAreaRef={scrollAreaRef}
            onSubmit={handleSubmit}
            onCancel={handleCancel}
            liveActivityEvents={processedEventsTimeline}
            historicalActivities={historicalActivities}
          />
        )}
      </main>
    </div>
  );
}
