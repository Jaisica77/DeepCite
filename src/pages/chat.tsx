import { useState, useRef, useEffect } from "react";
import { Send, FileText, Globe, Loader2, BookOpen, ToggleLeft, ToggleRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Navigation } from "@/components/Navigation";
import { agentApi, uploadApi } from "@/lib/api";
import type { Citation, ChatMessage } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  toolCallsMade?: number;
  loading?: boolean;
}

function renderWithCitations(text: string) {
  const parts = text.split(/(\[(?:DOC|WEB|Source \d+):[^\]]*\])/g);
  return parts.map((part, i) => {
    const docMatch = part.match(/\[DOC: ([^\]]+)\]/);
    const webMatch = part.match(/\[WEB: ([^\]]+)\]/);
    const srcMatch = part.match(/\[Source (\d+)\]/);

    if (docMatch) {
      return (
        <span
          key={i}
          className="inline-flex items-center gap-1 mx-0.5 px-1.5 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800 border border-green-200"
          title={docMatch[1]}
        >
          <FileText size={10} />
          {docMatch[1].length > 22 ? docMatch[1].slice(0, 22) + "…" : docMatch[1]}
        </span>
      );
    }
    if (webMatch) {
      let domain = webMatch[1];
      try { domain = new URL(webMatch[1]).hostname.replace("www.", ""); } catch {}
      return (
        <a
          key={i}
          href={webMatch[1]}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 mx-0.5 px-1.5 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800 border border-blue-200 hover:bg-blue-200 transition-colors"
          title={webMatch[1]}
        >
          <Globe size={10} />
          {domain}
        </a>
      );
    }
    if (srcMatch) {
      return (
        <sup key={i} className="text-primary font-bold ml-0.5 cursor-default" title={`Source ${srcMatch[1]}`}>
          [{srcMatch[1]}]
        </sup>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

function CitationCard({ citation, index }: { citation: Citation; index: number }) {
  const isDoc = citation.type === "doc";
  return (
    <div className={`rounded-lg border p-3 text-sm ${isDoc ? "border-green-200 bg-green-50" : "border-blue-200 bg-blue-50"}`}>
      <div className="flex items-center gap-2 mb-1.5">
        {isDoc
          ? <FileText size={13} className="text-green-700 shrink-0" />
          : <Globe size={13} className="text-blue-700 shrink-0" />}
        <span className={`font-semibold text-xs truncate ${isDoc ? "text-green-800" : "text-blue-800"}`}>
          [{index + 1}] {isDoc ? citation.source : (citation.title || citation.source)}
        </span>
      </div>
      <p className="text-gray-600 text-xs leading-relaxed line-clamp-3">{citation.snippet}</p>
      {!isDoc && (
        <a
          href={citation.source}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 text-xs hover:underline mt-1.5 block truncate"
        >
          {citation.source}
        </a>
      )}
    </div>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [useWeb, setUseWeb] = useState(true);
  const [activeCitations, setActiveCitations] = useState<Citation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const { data: docsData } = useQuery({
    queryKey: ["docs"],
    queryFn: () => uploadApi.listDocuments(),
  });
  const docs = docsData || [];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;
    const userMsg = input.trim();
    setInput("");
    setIsLoading(true);

    const history: ChatMessage[] = messages
      .filter(m => !m.loading)
      .map(m => ({ role: m.role, content: m.content }));

    setMessages(prev => [
      ...prev,
      { role: "user", content: userMsg },
      { role: "assistant", content: "", loading: true }
    ]);

    try {
      const data = await agentApi.chat({
        query: userMsg,
        doc_ids: selectedDocs,
        chat_history: history,
        use_web_search: useWeb,
      });

      setMessages(prev => [
        ...prev.slice(0, -1),
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations,
          toolCallsMade: data.tool_calls_made,
        }
      ]);
      setActiveCitations(data.citations || []);
    } catch (err: any) {
      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: "assistant", content: `Error: ${err.message || "Something went wrong. Please try again."}` }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleDoc = (doc_id: string) => {
    setSelectedDocs(prev =>
      prev.includes(doc_id) ? prev.filter(d => d !== doc_id) : [...prev, doc_id]
    );
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navigation />

      <div className="flex flex-1 pt-16 overflow-hidden h-[calc(100vh-4rem)]">

        <div className="flex flex-col flex-1 min-w-0">

          <div className="border-b px-6 py-3 flex items-center justify-between bg-background/80 backdrop-blur">
            <div className="flex items-center gap-2">
              <BookOpen size={18} className="text-primary" />
              <span className="font-semibold">DeepCite Chat</span>
              <span className="text-xs text-muted-foreground ml-1">· Every claim cited automatically</span>
            </div>
            <button
              onClick={() => setUseWeb(v => !v)}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border transition-colors ${
                useWeb
                  ? "bg-blue-50 border-blue-200 text-blue-700"
                  : "bg-muted border-border text-muted-foreground"
              }`}
            >
              {useWeb ? <ToggleRight size={14} /> : <ToggleLeft size={14} />}
              Web search {useWeb ? "on" : "off"}
            </button>
          </div>

          {docs.length > 0 && (
            <div className="border-b px-6 py-2 flex flex-wrap gap-2 bg-muted/30">
              <span className="text-xs text-muted-foreground self-center">Search in:</span>
              <button
                onClick={() => setSelectedDocs([])}
                className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                  selectedDocs.length === 0
                    ? "bg-primary text-primary-foreground border-primary"
                    : "bg-background border-border text-muted-foreground hover:border-primary"
                }`}
              >
                All docs
              </button>
              {docs.map(doc => (
                <button
                  key={doc.doc_id}
                  onClick={() => toggleDoc(doc.doc_id)}
                  className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                    selectedDocs.includes(doc.doc_id)
                      ? "bg-green-600 text-white border-green-600"
                      : "bg-background border-border text-muted-foreground hover:border-green-400"
                  }`}
                >
                  {doc.filename}
                </button>
              ))}
            </div>
          )}

          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground gap-4">
                <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center opacity-80">
                  <BookOpen size={32} className="text-white" />
                </div>
                <div>
                  <p className="font-semibold text-foreground">Ask DeepCite anything</p>
                  <p className="text-sm mt-1">
                    I'll search your documents and the web,<br />then cite every claim in my answer.
                  </p>
                </div>
                <div className="flex gap-2 flex-wrap justify-center mt-2">
                  {["What are the key findings?", "Summarise this document", "Compare with current research"].map(s => (
                    <button
                      key={s}
                      onClick={() => setInput(s)}
                      className="text-xs px-3 py-1.5 rounded-full border border-border hover:border-primary hover:text-primary transition-colors"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-card border shadow-sm text-foreground"
                  }`}
                >
                  {msg.loading ? (
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Loader2 size={14} className="animate-spin" />
                      Searching and thinking…
                    </div>
                  ) : msg.role === "assistant" && msg.citations ? (
                    <>
                      <div className="leading-7">{renderWithCitations(msg.content)}</div>
                      <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/50">
                        <span className="text-xs text-muted-foreground">
                          {msg.toolCallsMade} tool call{msg.toolCallsMade !== 1 ? "s" : ""}
                          {msg.citations.length > 0 && ` · ${msg.citations.length} source${msg.citations.length !== 1 ? "s" : ""}`}
                        </span>
                        {msg.citations.length > 0 && (
                          <button
                            onClick={() => setActiveCitations(msg.citations!)}
                            className="text-xs text-primary hover:underline"
                          >
                            View sources →
                          </button>
                        )}
                      </div>
                    </>
                  ) : (
                    msg.content
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          <div className="border-t px-6 py-4 bg-background">
            <div className="flex gap-2 items-end max-w-4xl mx-auto">
              <Textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
                className="resize-none min-h-[44px] max-h-[140px]"
                rows={1}
                disabled={isLoading}
              />
              <Button
                onClick={sendMessage}
                disabled={!input.trim() || isLoading}
                size="icon"
                className="shrink-0"
              >
                {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
              </Button>
            </div>
          </div>
        </div>

        <div className="w-72 border-l bg-muted/20 flex flex-col overflow-hidden shrink-0">
          <div className="px-4 py-3 border-b bg-background">
            <h2 className="font-semibold text-sm">Sources</h2>
            <div className="flex gap-3 mt-1">
              <span className="flex items-center gap-1 text-xs text-green-700">
                <FileText size={10} /> Your docs
              </span>
              <span className="flex items-center gap-1 text-xs text-blue-700">
                <Globe size={10} /> Web
              </span>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {activeCitations.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center mt-10 leading-relaxed">
                Sources from the latest<br />answer appear here
              </p>
            ) : (
              activeCitations.map((c, i) => (
                <CitationCard key={i} citation={c} index={i} />
              ))
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
