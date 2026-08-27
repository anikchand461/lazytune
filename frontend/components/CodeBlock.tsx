"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";
import { highlightLine, type Lang } from "@/lib/highlight";

export default function CodeBlock({
  code,
  lang = "python",
}: {
  code: string;
  lang?: Lang | "python" | "bash" | "signature";
}) {
  const [copied, setCopied] = useState(false);
  const lines = code.split("\n");

  function copy() {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="overflow-hidden rounded-md border border-border bg-[#0a0a0b]">
      <div className="flex items-center justify-between border-b border-border px-4 py-2">
        <span className="mono-label text-[10px] text-muted">{lang}</span>
        <button
          onClick={copy}
          className="mono-label flex items-center gap-1.5 text-[10px] text-muted transition-colors hover:text-accent"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto px-4 py-4 font-mono text-[13px] leading-relaxed">
        <code>
          {lines.map((line, i) => (
            <div key={i} className="flex">
              <span className="mr-4 w-5 flex-shrink-0 select-none text-right text-muted2/70">
                {line.trim() ? i + 1 : ""}
              </span>
              <span className="whitespace-pre">
                {line.length === 0
                  ? "\u00A0"
                  : highlightLine(line, lang as Lang).map((tok, j) => (
                      <span key={j} className={tok.cls}>
                        {tok.text}
                      </span>
                    ))}
              </span>
            </div>
          ))}
        </code>
      </pre>
    </div>
  );
}
