"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Header, { type Status } from "@/components/Header";
import Stepper from "@/components/Stepper";
import UploadCard from "@/components/UploadCard";
import ModelCard from "@/components/ModelCard";
import ParamsCard from "@/components/ParamsCard";
import ResultsCard, { type RunState } from "@/components/ResultsCard";
import { useToast } from "@/components/Toast";
import {
  MODEL_METRICS,
  newParamRow,
  parseValueList,
  type ModelName,
  type ParamRow,
  type ParamValue,
} from "@/lib/modelData";
import { runOptimization } from "@/lib/api";

const SECTION_IDS = ["section-dataset", "section-model", "section-params", "section-results"];

export default function Home() {
  const [step, setStep] = useState(1);
  const [status, setStatus] = useState<Status>("ready");

  const [model, setModelState] = useState<ModelName>("RandomForestClassifier");
  const [target, setTarget] = useState("");
  const [metric, setMetric] = useState(MODEL_METRICS.RandomForestClassifier[0]);
  const [rows, setRows] = useState<ParamRow[]>([newParamRow("RandomForestClassifier")]);

  const [runState, setRunState] = useState<RunState>("idle");
  const [output, setOutput] = useState("");

  const toast = useToast();

  function setModel(m: ModelName) {
    setModelState(m);
    setMetric(MODEL_METRICS[m][0]);
    setRows([newParamRow(m)]);
  }

  function jumpTo(n: number) {
    setStep(n);
    document.getElementById(SECTION_IDS[n - 1])?.scrollIntoView({ behavior: "smooth" });
  }

  function handleUploaded() {
    jumpTo(2);
  }

  async function handleRun() {
    const grid: Record<string, ParamValue[]> = {};
    rows.forEach((r) => {
      const values = parseValueList(r.value);
      if (values.length) grid[r.key] = values;
    });

    setStep(4);
    document.getElementById("section-results")?.scrollIntoView({ behavior: "smooth" });
    setRunState("running");
    setStatus("running");

    try {
      const data = await runOptimization({ model, target, metric, param_grid: grid });
      setOutput(JSON.stringify(data, null, 2));
      setRunState("done");
      setStatus("done");
      toast("optimization complete");
    } catch {
      setOutput("Backend error — is the server running?");
      setRunState("error");
      setStatus("error");
      toast("optimization failed", "err");
    }
  }

  return (
    <>
      <Header status={status} />
      <Stepper current={step} onJump={jumpTo} />

      <main className="relative z-10 mx-auto flex max-w-[880px] flex-col gap-8 px-6 pb-28 pt-14 sm:px-10">
        <Hero />
        <UploadCard onUploaded={handleUploaded} />
        <ModelCard
          model={model}
          setModel={setModel}
          target={target}
          setTarget={setTarget}
          metric={metric}
          setMetric={setMetric}
        />
        <ParamsCard
          model={model}
          rows={rows}
          setRows={setRows}
          onRun={handleRun}
          running={runState === "running"}
        />
        <ResultsCard state={runState} output={output} />
      </main>

      <Footer />
    </>
  );
}

function Hero() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut", delay: 0.1 }}
      className="mb-2"
    >
      <div className="mono-label mb-4 flex items-center gap-2 text-[11px] text-muted">
        <span className="h-1.5 w-1.5 rounded-full bg-accent" />
        Screen &rarr; Prune &rarr; Train
      </div>
      <h1 className="glow-text font-display text-[54px] leading-[0.95] tracking-wide text-text sm:text-[72px]">
        Stop brute-forcing
        <br />
        <span className="text-accent">GridSearchCV.</span>
      </h1>
      <p className="mt-5 max-w-lg text-[16px] leading-relaxed text-muted">
        Upload a dataset, choose a model, and hand LazyTune a parameter grid.
        It screens every candidate, prunes the weak ones, and fully trains
        only the survivors — four steps, one run.
      </p>
    </motion.div>
  );
}

function Footer() {
  return (
    <footer className="relative z-10 border-t border-border/60 py-10">
      <div className="mx-auto flex max-w-[1200px] flex-col items-center gap-2 px-6 text-center sm:px-10">
        <div className="mono-label text-[10px] text-muted2">
          LazyTune &mdash; a fast screening &rarr; pruning &rarr; full-training
          pipeline for scikit-learn.
        </div>
        <div className="mono-label text-[10px] text-muted2">
          MIT License &middot; Built by Anik Chand
        </div>
      </div>
    </footer>
  );
}
