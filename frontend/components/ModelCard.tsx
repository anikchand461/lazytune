"use client";

import SectionCard from "./SectionCard";
import { SelectField, TextField } from "./FormControls";
import { MODEL_METRICS, MODEL_NAMES, type ModelName } from "@/lib/modelData";

export default function ModelCard({
  model,
  setModel,
  target,
  setTarget,
  metric,
  setMetric,
}: {
  model: ModelName;
  setModel: (m: ModelName) => void;
  target: string;
  setTarget: (t: string) => void;
  metric: string;
  setMetric: (m: string) => void;
}) {
  return (
    <SectionCard
      id="section-model"
      index="02"
      label="Model"
      title="Configure Model"
      description="Select the algorithm, target column, and the metric to optimize."
    >
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        <div>
          <label className="mono-label mb-2 block text-[10px] text-muted">
            Model
          </label>
          <SelectField
            value={model}
            onChange={(v) => setModel(v as ModelName)}
            options={MODEL_NAMES}
            ariaLabel="Model"
          />
        </div>
        <div>
          <label className="mono-label mb-2 block text-[10px] text-muted">
            Target Column
          </label>
          <TextField
            value={target}
            onChange={setTarget}
            placeholder="e.g. target"
            ariaLabel="Target column"
          />
        </div>
        <div>
          <label className="mono-label mb-2 block text-[10px] text-muted">
            Scoring Metric
          </label>
          <SelectField
            value={metric}
            onChange={setMetric}
            options={MODEL_METRICS[model]}
            ariaLabel="Scoring metric"
          />
        </div>
      </div>
    </SectionCard>
  );
}
