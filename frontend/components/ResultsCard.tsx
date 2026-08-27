"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Copy, Check, TerminalSquare } from "lucide-react";
import { useState } from "react";
import SectionCard from "./SectionCard";
import ScanGrid from "./ScanGrid";
import { useToast } from "./Toast";

export type RunState = "idle" | "running" | "done" | "error";

export default function ResultsCard({
  state,
  output,
}: {
  state: RunState;
  output: string;
}) {
  const [copied, setCopied] = useState(false);
  const toast = useToast();

  function copy() {
    navigator.clipboard.writeText(output);
    setCopied(true);
    toast("copied to clipboard");
    setTimeout(() => setCopied(false), 1600);
  }

  return (
    <SectionCard
      id="section-results"
      index="04"
      label="Results"
      title="Output"
      description="Best parameters and score will appear here after the run completes."
    >
      <div
        className={`overflow-hidden rounded-md border p-5 transition-colors duration-300 ${
          state === "error"
            ? "border-danger/40 bg-[#160a0a]"
            : state === "done"
              ? "border-accent/30 bg-surface"
              : "border-border bg-surface"
        }`}
      >
        <AnimatePresence mode="wait">
          {state === "idle" && (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-2 py-6 font-mono text-[13px] text-muted"
            >
              <TerminalSquare size={16} />
              Awaiting optimization...
            </motion.div>
          )}

          {state === "running" && (
            <motion.div
              key="running"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col gap-4 py-2"
            >
              <div className="mono-label flex items-center gap-2 text-[11px] text-accent2">
                <span className="h-1.5 w-1.5 animate-blink rounded-full bg-accent2" />
                Screening candidates &rarr; pruning &rarr; training survivors
              </div>
              <ScanGrid active />
            </motion.div>
          )}

          {(state === "done" || state === "error") && (
            <motion.pre
              key="result"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={`whitespace-pre-wrap break-words font-mono text-[13px] leading-relaxed ${
                state === "error" ? "text-danger" : "text-accent"
              }`}
            >
              {output}
            </motion.pre>
          )}
        </AnimatePresence>
      </div>

      {state === "done" && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={copy}
          className="mono-label mt-5 inline-flex items-center gap-2 rounded-md border border-border px-4 py-2.5 text-[11px] text-muted transition-colors hover:border-accent hover:text-accent"
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
          {copied ? "Copied" : "Copy Result"}
        </motion.button>
      )}
    </SectionCard>
  );
}
