"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

export default function Loader() {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const t = setTimeout(() => setVisible(false), 1400);
    return () => clearTimeout(t);
  }, []);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
          className="fixed inset-0 z-[99999] flex flex-col items-center justify-center gap-7 bg-bg"
        >
          <motion.svg
            viewBox="0 0 80 80"
            className="h-16 w-16"
            style={{ filter: "drop-shadow(0 0 18px #c8ff00)" }}
            animate={{ rotate: 360, scale: [1, 1.08, 1, 1.08, 1] }}
            transition={{
              rotate: { duration: 2, repeat: Infinity, ease: "linear" },
              scale: { duration: 2, repeat: Infinity, ease: "easeInOut" },
            }}
          >
            <polygon
              points="40,3 74,21 74,59 40,77 6,59 6,21"
              fill="#c8ff00"
            />
            <path d="M46 12 L27 44 H39 L34 68 L57 32 H45 Z" fill="#0a0a0b" />
          </motion.svg>

          <div className="h-[2px] w-48 overflow-hidden rounded-full bg-border">
            <motion.div
              initial={{ width: "0%" }}
              animate={{ width: "100%" }}
              transition={{ duration: 1.1, ease: "easeInOut" }}
              className="h-full bg-gradient-to-r from-accent to-accent2"
              style={{ boxShadow: "0 0 10px #c8ff00" }}
            />
          </div>

          <motion.div
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
            className="mono-label text-[11px] text-muted"
          >
            Initializing LazyTune
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
