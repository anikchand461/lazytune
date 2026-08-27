"use client";

import { useEffect, useRef, useState } from "react";

const COLS = 12;
const ROWS = 5;
const TOTAL = COLS * ROWS;

type CellState = "idle" | "screening" | "pruned" | "survivor";

export default function ScanGrid({ active }: { active: boolean }) {
  const [cells, setCells] = useState<CellState[]>(
    Array.from({ length: TOTAL }, () => "idle"),
  );
  const tick = useRef(0);

  useEffect(() => {
    if (!active) {
      setCells(Array.from({ length: TOTAL }, () => "idle"));
      tick.current = 0;
      return;
    }
    const id = setInterval(() => {
      tick.current += 1;
      setCells((prev) =>
        prev.map((c, i) => {
          if (tick.current < 10) {
            // screening phase: cells light up in a rolling wave
            const wave = (tick.current * 7) % TOTAL;
            const dist = Math.min(
              Math.abs(i - wave),
              TOTAL - Math.abs(i - wave),
            );
            return dist < 4 ? "screening" : c === "screening" ? "idle" : c;
          }
          // prune & train phase: settle each cell permanently
          if (c === "screening" || c === "idle") {
            return Math.random() < 0.28 ? "survivor" : "pruned";
          }
          return c;
        }),
      );
    }, 90);
    return () => clearInterval(id);
  }, [active]);

  return (
    <div
      className="grid gap-1.5"
      style={{ gridTemplateColumns: `repeat(${COLS}, minmax(0,1fr))` }}
      aria-hidden
    >
      {cells.map((c, i) => (
        <div
          key={i}
          className="aspect-square rounded-[2px] transition-all duration-300 ease-out"
          style={{
            background:
              c === "survivor"
                ? "#c8ff00"
                : c === "screening"
                  ? "#00e5ff"
                  : c === "pruned"
                    ? "#1c1d20"
                    : "#17181b",
            boxShadow:
              c === "survivor"
                ? "0 0 8px rgba(200,255,0,.7)"
                : c === "screening"
                  ? "0 0 8px rgba(0,229,255,.7)"
                  : "none",
            opacity: c === "pruned" ? 0.35 : 1,
          }}
        />
      ))}
    </div>
  );
}
