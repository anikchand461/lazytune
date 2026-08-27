"use client";

import { createContext, useCallback, useContext, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";

type ToastType = "ok" | "err";
interface ToastState {
  id: number;
  message: string;
  type: ToastType;
}

const ToastCtx = createContext<(message: string, type?: ToastType) => void>(
  () => {},
);

export function useToast() {
  return useContext(ToastCtx);
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastState[]>([]);
  const idRef = useRef(0);

  const show = useCallback((message: string, type: ToastType = "ok") => {
    const id = ++idRef.current;
    setToasts((t) => [...t, { id, message, type }]);
    setTimeout(() => {
      setToasts((t) => t.filter((x) => x.id !== id));
    }, 3000);
  }, []);

  return (
    <ToastCtx.Provider value={show}>
      {children}
      <div className="fixed bottom-6 left-1/2 z-[9999] flex -translate-x-1/2 flex-col items-center gap-2">
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, y: 16, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.96 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              className={`mono-label flex items-center gap-2 rounded-md border px-4 py-3 text-[11px] shadow-lg backdrop-blur-md ${
                t.type === "ok"
                  ? "border-accent/40 bg-[#0f1400]/95 text-accent"
                  : "border-danger/40 bg-[#180a0a]/95 text-danger"
              }`}
            >
              {t.type === "ok" ? (
                <CheckCircle2 size={14} strokeWidth={2.5} />
              ) : (
                <XCircle size={14} strokeWidth={2.5} />
              )}
              {t.message}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastCtx.Provider>
  );
}
