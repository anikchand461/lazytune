"use client";

import { motion } from "framer-motion";

export default function SectionCard({
  id,
  index,
  label,
  title,
  description,
  children,
}: {
  id: string;
  index: string;
  label: string;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.55, ease: "easeOut" }}
      className="bracket scroll-mt-40 rounded-lg border border-border bg-card/70 p-6 backdrop-blur-sm sm:p-9"
    >
      <div className="mono-label mb-2 text-[11px] text-accent">
        {index} &mdash; {label}
      </div>
      <h2 className="font-display text-4xl tracking-wide text-text sm:text-[42px]">
        {title}
      </h2>
      <p className="mt-2 max-w-xl text-[15px] leading-relaxed text-muted">
        {description}
      </p>
      <div className="mt-7">{children}</div>
    </motion.section>
  );
}
