"use client";

import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { UploadCloud, FileSpreadsheet, Loader2 } from "lucide-react";
import SectionCard from "./SectionCard";
import { useToast } from "./Toast";
import { uploadDataset } from "@/lib/api";

export default function UploadCard({
  onUploaded,
}: {
  onUploaded: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  function pick(f: File | undefined | null) {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".csv")) {
      toast("please choose a .csv file", "err");
      return;
    }
    setFile(f);
  }

  async function handleUpload() {
    if (!file) {
      toast("select a file first", "err");
      return;
    }
    setBusy(true);
    try {
      await uploadDataset(file);
      toast("dataset uploaded");
      onUploaded();
    } catch {
      toast("upload failed — is the server running?", "err");
    } finally {
      setBusy(false);
    }
  }

  return (
    <SectionCard
      id="section-dataset"
      index="01"
      label="Dataset"
      title="Upload CSV"
      description="Drop a dataset file to begin. Accepts standard CSV format."
    >
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files[0]);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-md border-2 border-dashed px-6 py-14 text-center transition-colors duration-200 ${
          dragging
            ? "border-accent bg-accent/5"
            : "border-border bg-surface/60 hover:border-borderHi"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />
        <motion.div
          animate={dragging ? { y: -4 } : { y: 0 }}
          className={`flex h-14 w-14 items-center justify-center rounded-full border ${
            dragging ? "border-accent text-accent" : "border-border text-muted"
          }`}
        >
          {file ? <FileSpreadsheet size={22} /> : <UploadCloud size={22} />}
        </motion.div>
        <div className="text-[15px] font-medium text-text">
          Click or drag to upload
        </div>
        <div className="mono-label text-[11px] text-accent">
          {file ? file.name : "No file chosen"}
        </div>
      </div>

      <button
        onClick={handleUpload}
        disabled={busy}
        className="mono-label mt-6 inline-flex items-center gap-2 rounded-md bg-accent px-6 py-3 text-[12px] font-semibold text-bg transition-transform duration-150 hover:-translate-y-0.5 hover:shadow-[0_0_24px_rgba(200,255,0,.4)] active:translate-y-0 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {busy ? (
          <Loader2 size={15} className="animate-spin" />
        ) : (
          <UploadCloud size={15} />
        )}
        {busy ? "Uploading..." : "Upload Dataset"}
      </button>
    </SectionCard>
  );
}
