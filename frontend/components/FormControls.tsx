"use client";

import { ChevronDown } from "lucide-react";

export function SelectField({
  value,
  onChange,
  options,
  className = "",
  ariaLabel,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <div className={`relative ${className}`}>
      <select
        value={value}
        aria-label={ariaLabel}
        onChange={(e) => onChange(e.target.value)}
        className="w-full appearance-none rounded-md border border-border bg-surface px-4 py-3 pr-10 font-mono text-[14px] text-text transition-colors focus:border-accent"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
      <ChevronDown
        size={16}
        className="pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2 text-muted"
      />
    </div>
  );
}

export function TextField({
  value,
  onChange,
  placeholder,
  className = "",
  ariaLabel,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <input
      type="text"
      value={value}
      aria-label={ariaLabel}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className={`w-full rounded-md border border-border bg-surface px-4 py-3 font-mono text-[14px] text-text placeholder:text-muted2 transition-colors focus:border-accent ${className}`}
    />
  );
}
