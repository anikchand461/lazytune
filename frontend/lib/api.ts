import type { ModelName, ParamValue } from "./modelData";

// Override with NEXT_PUBLIC_API_BASE in .env.local to point at a different backend.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "https://lazytune.onrender.com";

export async function uploadDataset(file: File): Promise<void> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/datasets/upload`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(`Upload failed (${res.status})`);
}

export interface OptimizePayload {
  model: ModelName;
  target: string;
  metric: string;
  param_grid: Record<string, ParamValue[]>;
}

export async function runOptimization(payload: OptimizePayload): Promise<unknown> {
  const res = await fetch(`${API_BASE}/optimize/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Optimization failed (${res.status})`);
  return res.json();
}
