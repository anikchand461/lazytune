"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";

export type Status = "ready" | "running" | "done" | "error";

const STATUS_META: Record<Status, { label: string; color: string; glow: string }> = {
  ready: { label: "Ready", color: "#c8ff00", glow: "rgba(200,255,0,.6)" },
  running: { label: "Running", color: "#facc15", glow: "rgba(250,204,21,.6)" },
  done: { label: "Done", color: "#00e5ff", glow: "rgba(0,229,255,.6)" },
  error: { label: "Error", color: "#ff4d4d", glow: "rgba(255,77,77,.6)" },
};

export default function Header({ status = "ready" }: { status?: Status }) {
  const pathname = usePathname();
  const meta = STATUS_META[status];

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-bg/90 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1200px] items-center justify-between gap-3 px-6 py-3 sm:px-10">
        <Link href="/" className="flex flex-shrink-0 items-center gap-3">
          <motion.svg
            viewBox="0 0 44 44"
            className="h-9 w-9 flex-shrink-0"
            animate={{
              filter: [
                "drop-shadow(0 0 4px #c8ff00)",
                "drop-shadow(0 0 14px #c8ff00)",
                "drop-shadow(0 0 4px #c8ff00)",
              ],
            }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          >
            <polygon
              points="22,2 40,11.5 40,32.5 22,42 4,32.5 4,11.5"
              fill="#c8ff00"
            />
            <path d="M25 8 L15 24 H22 L19 36 L31 18 H24 Z" fill="#0a0a0b" />
          </motion.svg>
          <div className="leading-none">
            <div className="font-display text-[26px] tracking-wide text-accent">
              LazyTune
            </div>
            <div className="mono-label mt-0.5 text-[9px] text-muted">
              Hyperparameter Optimizer
            </div>
          </div>
        </Link>

        <div className="flex items-center gap-2 sm:gap-4">
          <nav className="flex items-center gap-1">
            <NavLink href="/" active={pathname === "/"}>
              App
            </NavLink>
            <NavLink href="/docs" active={pathname === "/docs"}>
              Docs
            </NavLink>
          </nav>
          <div
            className="mono-label flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-[11px] text-text/90"
            aria-live="polite"
          >
            <span
              className="h-1.5 w-1.5 rounded-full transition-colors duration-300"
              style={{ background: meta.color, boxShadow: `0 0 10px ${meta.glow}` }}
            />
            {meta.label}
          </div>
        </div>
      </div>
    </header>
  );
}

function NavLink({
  href,
  active,
  children,
}: {
  href: string;
  active: boolean;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className={`mono-label rounded-md border px-3.5 py-1.5 text-[11px] transition-colors ${
        active
          ? "border-accent bg-accent text-bg"
          : "border-transparent text-muted hover:border-border hover:text-accent"
      }`}
    >
      {children}
    </Link>
  );
}
