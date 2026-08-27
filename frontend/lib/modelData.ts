export type ModelName =
  | "RandomForestClassifier"
  | "SVC"
  | "LogisticRegression"
  | "RandomForestRegressor"
  | "LinearRegression";

export const MODEL_NAMES: ModelName[] = [
  "RandomForestClassifier",
  "SVC",
  "LogisticRegression",
  "RandomForestRegressor",
  "LinearRegression",
];

export const MODEL_METRICS: Record<ModelName, string[]> = {
  RandomForestClassifier: ["accuracy", "f1", "precision", "recall"],
  SVC: ["accuracy", "f1", "precision", "recall"],
  LogisticRegression: ["accuracy", "f1", "precision", "recall"],
  RandomForestRegressor: ["r2", "neg_mean_squared_error"],
  LinearRegression: ["r2", "neg_mean_squared_error"],
};

export const MODEL_PARAMS: Record<ModelName, string[]> = {
  RandomForestClassifier: [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "bootstrap",
    "criterion",
  ],
  RandomForestRegressor: [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "bootstrap",
    "criterion",
  ],
  SVC: ["C", "kernel", "degree", "gamma", "coef0", "shrinking"],
  LogisticRegression: [
    "C",
    "solver",
    "penalty",
    "max_iter",
    "fit_intercept",
    "class_weight",
  ],
  LinearRegression: ["fit_intercept", "copy_X", "positive"],
};

export const PARAM_HINTS: Record<string, string> = {
  n_estimators: "example: 50,100,200",
  max_depth: "example: 5,10,20 or null",
  min_samples_split: "example: 2,5,10",
  min_samples_leaf: "example: 1,2,4",
  max_features: "options: sqrt,log2 or number",
  bootstrap: "options: true,false",
  criterion:
    "classifier: gini,entropy | regressor: squared_error,absolute_error",
  C: "example: 0.01,0.1,1,10",
  kernel: "options: linear,rbf,poly,sigmoid",
  degree: "example: 2,3,4 (poly kernel)",
  gamma: "options: scale,auto or 0.1",
  coef0: "example: 0,0.1,0.5",
  shrinking: "options: true,false",
  solver: "options: lbfgs,liblinear,newton-cg,saga",
  penalty: "options: l1,l2,elasticnet,none",
  max_iter: "example: 100,200,500",
  fit_intercept: "options: true,false",
  class_weight: "options: balanced or none",
  copy_X: "options: true,false",
  positive: "options: true,false",
};

export interface ParamRow {
  id: string;
  key: string;
  value: string;
}

let rowSeq = 0;
export function newParamRow(model: ModelName): ParamRow {
  rowSeq += 1;
  return { id: `p${rowSeq}-${Date.now()}`, key: MODEL_PARAMS[model][0], value: "" };
}

export type ParamValue = string | number | boolean;

export function parseValueList(raw: string): ParamValue[] {
  return raw
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean)
    .map((v): ParamValue => {
      if (v === "true") return true;
      if (v === "false") return false;
      return isNaN(Number(v)) ? v : Number(v);
    });
}
