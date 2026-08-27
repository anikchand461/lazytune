"use client";

import { motion } from "framer-motion";

const STEPS = [
  { n: 1, label: "Dataset" },
  { n: 2, label: "Model" },
  { n: 3, label: "Parameters" },
  { n: 4, label: "Results" },
];

export default function Stepper({
  current,
  onJump,
}: {
  current: number;
  onJump: (n: number) => void;
}) {
  return (
    <div className="sticky top-[65px] z-40 border-b border-border/60 bg-bg/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1200px] items-center gap-0 overflow-x-auto px-6 py-4 sm:px-10">
        {STEPS.map((s, i) => {
          const state =
            s.n === current ? "active" : s.n < current ? "done" : "idle";
          return (
            <div key={s.n} className="flex items-center">
              <button
                onClick={() => onJump(s.n)}
                className="group flex items-center gap-2.5 rounded-md px-1.5 py-1 transition-opacity"
              >
                <span
                  className={`mono-label flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full border text-[11px] transition-all duration-300 ${
                    state === "active"
                      ? "border-accent bg-accent text-bg shadow-[0_0_16px_rgba(200,255,0,.55)]"
                      : state === "done"
                        ? "border-accent/50 bg-transparent text-accent"
                        : "border-border text-muted"
                  }`}
                >
                  {state === "done" ? "✓" : s.n}
                </span>
                <span
                  className={`mono-label text-[11px] transition-colors ${
                    state === "active"
                      ? "text-text"
                      : state === "done"
                        ? "text-muted2"
                        : "text-muted2"
                  } group-hover:text-accent`}
                >
                  {s.label}
                </span>
              </button>
              {i < STEPS.length - 1 && (
                <div className="mx-2 h-px w-8 overflow-hidden bg-border sm:w-16">
                  <motion.div
                    className="h-full bg-accent"
                    initial={false}
                    animate={{ width: s.n < current ? "100%" : "0%" }}
                    transition={{ duration: 0.45, ease: "easeInOut" }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
