"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Plus, X, Play, Loader2 } from "lucide-react";
import SectionCard from "./SectionCard";
import { SelectField, TextField } from "./FormControls";
import {
  MODEL_PARAMS,
  PARAM_HINTS,
  newParamRow,
  type ModelName,
  type ParamRow,
} from "@/lib/modelData";

export default function ParamsCard({
  model,
  rows,
  setRows,
  onRun,
  running,
}: {
  model: ModelName;
  rows: ParamRow[];
  setRows: (rows: ParamRow[]) => void;
  onRun: () => void;
  running: boolean;
}) {
  function addRow() {
    setRows([...rows, newParamRow(model)]);
  }
  function removeRow(id: string) {
    setRows(rows.filter((r) => r.id !== id));
  }
  function updateRow(id: string, patch: Partial<ParamRow>) {
    setRows(rows.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  }

  return (
    <SectionCard
      id="section-params"
      index="03"
      label="Parameters"
      title="Parameter Grid"
      description="Add hyperparameters and their candidate values (comma-separated)."
    >
      <div className="flex flex-col gap-3">
        <AnimatePresence initial={false}>
          {rows.map((row) => (
            <motion.div
              key={row.id}
              initial={{ opacity: 0, height: 0, marginBottom: 0 }}
              animate={{ opacity: 1, height: "auto", marginBottom: 0 }}
              exit={{ opacity: 0, height: 0, marginBottom: 0 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              className="flex items-center gap-3 overflow-hidden"
            >
              <SelectField
                value={row.key}
                onChange={(v) => updateRow(row.id, { key: v })}
                options={MODEL_PARAMS[model]}
                className="w-[210px] flex-shrink-0"
                ariaLabel="Parameter name"
              />
              <TextField
                value={row.value}
                onChange={(v) => updateRow(row.id, { value: v })}
                placeholder={PARAM_HINTS[row.key] || "comma separated values"}
                ariaLabel="Parameter values"
              />
              <button
                onClick={() => removeRow(row.id)}
                aria-label="Remove parameter"
                className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-md border border-border text-muted transition-colors hover:border-danger hover:text-danger"
              >
                <X size={16} />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>

        <button
          onClick={addRow}
          className="mono-label inline-flex w-fit items-center gap-2 rounded-md border border-border px-4 py-2.5 text-[11px] text-muted transition-colors hover:border-accent hover:text-accent"
        >
          <Plus size={14} />
          Add Parameter
        </button>
      </div>

      <div className="my-7 h-px bg-border" />

      <motion.button
        whileHover={{ y: -2 }}
        whileTap={{ y: 0 }}
        onClick={onRun}
        disabled={running}
        className="mono-label flex w-full items-center justify-center gap-2 rounded-md bg-accent py-4 text-[13px] font-semibold text-bg shadow-[0_0_0_rgba(200,255,0,0)] transition-shadow duration-200 hover:shadow-[0_0_30px_rgba(200,255,0,.45)] disabled:cursor-not-allowed disabled:opacity-60"
      >
        {running ? (
          <Loader2 size={16} className="animate-spin" />
        ) : (
          <Play size={15} fill="currentColor" />
        )}
        {running ? "Running LazyTune..." : "Run LazyTune"}
      </motion.button>
    </SectionCard>
  );
}
