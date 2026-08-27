import Header from "@/components/Header";
import CodeBlock from "@/components/CodeBlock";
import { Github } from "lucide-react";

const NAV = [
  {
    group: "Getting Started",
    links: [
      { href: "#overview", label: "Overview" },
      { href: "#installation", label: "Installation" },
      { href: "#quick-start", label: "Quick Start" },
    ],
  },
  {
    group: "Examples",
    links: [
      { href: "#svm", label: "SVM Classification" },
      { href: "#regression", label: "Regression" },
    ],
  },
  {
    group: "Reference",
    links: [
      { href: "#metrics", label: "Supported Metrics" },
      { href: "#how-it-works", label: "How It Works" },
      { href: "#api", label: "API Reference" },
      { href: "#attributes", label: "Attributes" },
      { href: "#methods", label: "Methods" },
    ],
  },
  {
    group: "Project",
    links: [
      { href: "#requirements", label: "Requirements" },
      { href: "#license", label: "License" },
    ],
  },
];

const CLASSIFICATION_METRICS = [
  "accuracy",
  "f1",
  "f1_macro",
  "f1_weighted",
  "precision",
  "recall",
  "roc_auc",
  "balanced_accuracy",
];
const REGRESSION_METRICS = [
  "r2",
  "neg_mean_squared_error",
  "neg_root_mean_squared_error",
  "neg_mean_absolute_error",
  "neg_mean_absolute_percentage_error",
];

const PIPELINE = [
  {
    n: 1,
    title: "Generate Combinations",
    body: "All hyperparameter combinations are produced from the user-defined param_grid.",
  },
  {
    n: 2,
    title: "Screening Round",
    body: "Every candidate is quickly evaluated with cross-validation using minimal resources — just enough to rank relative performance.",
  },
  {
    n: 3,
    title: "Rank & Prune",
    body: "Candidates are sorted by screening score. The bottom prune_ratio fraction are eliminated before full training begins.",
  },
  {
    n: 4,
    title: "Full Training",
    body: "Only top-ranked survivors are trained fully. The best model, parameters, score and detailed trial summary are returned.",
  },
];

const ATTRIBUTES = [
  { name: "best_params_", type: "dict", desc: "Best found hyperparameter dictionary." },
  { name: "best_score_", type: "float", desc: "Best cross-validated score achieved." },
  { name: "best_estimator_", type: "estimator", desc: "Fully fitted estimator with best parameters." },
  { name: "summary_", type: "DataFrame", desc: "pandas DataFrame with all trial results and rankings." },
  { name: "cv_results_", type: "dict", desc: "Detailed cross-validation results per candidate." },
];

const METHODS = [
  { name: ".fit(X, y)", desc: "Run the full optimization pipeline on training data." },
  { name: ".predict(X)", desc: "Predict using the best found estimator." },
  { name: ".score(X, y)", desc: "Score the best estimator on given data." },
  { name: ".get_params()", desc: "Get parameters for this estimator." },
  { name: ".set_params(**params)", desc: "Set parameters of this estimator." },
];

const QUICK_START = `from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from lazytune import SmartSearch

X, y = load_breast_cancer(return_X_y=True)
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 3, 4, 5]
}

search = SmartSearch(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    metric="accuracy",
    cv_folds=3,
    prune_ratio=0.5,   # keep top 50% after screening
    n_jobs=-1          # use all available cores
)

search.fit(X, y)
print("Best parameters:", search.best_params_)
print("Best CV score:", search.best_score_)
print("\\nBest model:\\n", search.best_estimator_)`;

const SVM_EXAMPLE = `from sklearn.svm import SVC
from lazytune import SmartSearch

search = SmartSearch(
    estimator=SVC(random_state=42),
    param_grid={
        "C": [0.1, 1, 10, 50, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto", 0.001, 0.0001]
    },
    metric="f1_macro",
    cv_folds=5,
    prune_ratio=0.6
)`;

const REGRESSION_EXAMPLE = `from sklearn.ensemble import RandomForestRegressor
from lazytune import SmartSearch

search = SmartSearch(
    estimator=RandomForestRegressor(random_state=42),
    param_grid={
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 4, 8]
    },
    metric="r2",
    cv_folds=4,
    n_jobs=-1
)`;

const SIGNATURE = `SmartSearch(
    estimator,        # any scikit-learn style estimator
    param_grid,       # dict of param -> list of values
    metric,           # scoring string or make_scorer object
    cv_folds=3,        # number of CV folds for screening
    prune_ratio=0.5,   # fraction to prune (0.0 = keep all)
    n_jobs=1           # parallel workers (-1 = all cores)
)`;

export default function DocsPage() {
  return (
    <>
      <Header />
      <div className="relative z-10 mx-auto flex max-w-[1200px] gap-12 px-6 py-12 sm:px-10">
        <aside className="sticky top-24 hidden h-fit w-[220px] flex-shrink-0 lg:block">
          <div className="mono-label mb-4 text-[10px] text-muted">
            Documentation
          </div>
          <nav className="flex flex-col gap-6">
            {NAV.map((g) => (
              <div key={g.group}>
                <div className="mono-label mb-2 text-[10px] text-muted2">
                  {g.group}
                </div>
                <div className="flex flex-col gap-1.5">
                  {g.links.map((l) => (
                    <a
                      key={l.href}
                      href={l.href}
                      className="text-[13px] text-muted transition-colors hover:text-accent"
                    >
                      {l.label}
                    </a>
                  ))}
                </div>
              </div>
            ))}
          </nav>
        </aside>

        <main className="min-w-0 flex-1 pb-24">
          <div className="mono-label mb-3 inline-block rounded-full border border-border px-3 py-1 text-[10px] text-accent">
            v1.0 &mdash; Stable
          </div>
          <h1 className="font-display text-5xl tracking-wide sm:text-6xl">
            LazyTune Docs
          </h1>
          <p className="mt-4 max-w-2xl text-[16px] leading-relaxed text-muted">
            A fast and efficient hyperparameter optimization framework for
            scikit-learn models. Dramatically reduces training time with a
            smart screening &rarr; pruning &rarr; full-training pipeline.
          </p>
          <div className="mt-5 flex flex-wrap gap-2">
            {["Classification", "Regression", "scikit-learn compatible", "Python \u2265 3.8", "MIT License"].map(
              (chip) => (
                <span
                  key={chip}
                  className="mono-label rounded-full border border-border px-3 py-1 text-[10px] text-muted"
                >
                  {chip}
                </span>
              ),
            )}
          </div>

          <Section id="overview" index="Overview">
            <p className="text-[15px] leading-relaxed text-muted">
              LazyTune wraps any scikit-learn estimator and searches a
              parameter grid the way an experienced practitioner would: cheap
              screening first, then spend full training budget only on
              configurations that already look promising.
            </p>
          </Section>

          <Section id="installation" index="Installation">
            <p className="mb-4 text-[15px] leading-relaxed text-muted">
              Install LazyTune via pip. No extra configuration needed — all
              dependencies are pulled in automatically.
            </p>
            <CodeBlock lang="bash" code={`$ pip install lazytune`} />
            <p className="mt-3 text-[13px] text-muted2">
              Requires Python 3.8+, numpy, pandas, and scikit-learn.
            </p>
          </Section>

          <Section id="quick-start" index="Quick Start">
            <p className="mb-4 text-[15px] leading-relaxed text-muted">
              Get up and running with RandomForestClassifier on the breast
              cancer dataset in under a minute.
            </p>
            <CodeBlock code={QUICK_START} />
          </Section>

          <Section id="svm" index="SVM Classification">
            <p className="mb-4 text-[15px] leading-relaxed text-muted">
              Use SmartSearch with a Support Vector Machine to tune C,
              kernel, and gamma together.
            </p>
            <CodeBlock code={SVM_EXAMPLE} />
          </Section>

          <Section id="regression" index="Regression">
            <p className="mb-4 text-[15px] leading-relaxed text-muted">
              Works identically for regression — just switch the estimator
              and use a regression metric like r2.
            </p>
            <CodeBlock code={REGRESSION_EXAMPLE} />
          </Section>

          <Section id="metrics" index="Supported Metrics">
            <p className="mb-5 text-[15px] leading-relaxed text-muted">
              LazyTune supports all scikit-learn scoring strings. Pass any as
              the metric argument. For custom metrics use
              sklearn.metrics.make_scorer.
            </p>
            <div className="grid gap-6 sm:grid-cols-2">
              <MetricGroup title="Classification" items={CLASSIFICATION_METRICS} />
              <MetricGroup title="Regression" items={REGRESSION_METRICS} />
            </div>
          </Section>

          <Section id="how-it-works" index="How It Works">
            <p className="mb-6 text-[15px] leading-relaxed text-muted">
              LazyTune&rsquo;s four-phase pipeline eliminates wasted compute
              compared to brute-force GridSearchCV — while typically reaching
              identical final performance.
            </p>
            <div className="flex flex-col gap-4">
              {PIPELINE.map((p) => (
                <div
                  key={p.n}
                  className="bracket flex gap-4 rounded-md border border-border bg-card/60 p-5"
                >
                  <div className="font-display flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full border border-accent/40 text-[15px] text-accent">
                    {p.n}
                  </div>
                  <div>
                    <div className="mb-1 text-[15px] font-semibold text-text">
                      {p.title}
                    </div>
                    <div className="text-[14px] leading-relaxed text-muted">
                      {p.body}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section id="api" index="API Reference">
            <p className="mb-4 text-[15px] leading-relaxed text-muted">
              All functionality is exposed through the SmartSearch class.
            </p>
            <CodeBlock lang="signature" code={SIGNATURE} />
          </Section>

          <Section id="attributes" index="Attributes">
            <Table
              head={["Attribute", "Type", "Description"]}
              rows={ATTRIBUTES.map((a) => [
                <code key="n" className="text-accent">{a.name}</code>,
                a.type,
                a.desc,
              ])}
            />
          </Section>

          <Section id="methods" index="Methods">
            <Table
              head={["Method", "Description"]}
              rows={METHODS.map((m) => [
                <code key="n" className="text-accent">{m.name}</code>,
                m.desc,
              ])}
            />
          </Section>

          <Section id="requirements" index="Requirements">
            <ul className="flex flex-col gap-2 text-[15px] text-muted">
              {["Python \u2265 3.8", "numpy", "pandas", "scikit-learn"].map((r) => (
                <li key={r} className="flex items-center gap-2">
                  <span className="h-1 w-1 rounded-full bg-accent" />
                  {r}
                </li>
              ))}
            </ul>
            <p className="mt-4 text-[13px] text-muted2">
              All dependencies are installed automatically via pip.
            </p>
          </Section>

          <Section id="license" index="License & Author">
            <p className="mb-4 text-[15px] leading-relaxed text-muted">
              LazyTune is released under the MIT License — free to use,
              modify, and distribute. Built by Anik Chand. Feedback, issues,
              stars, and contributions are very welcome!
            </p>
            <a
              href="https://github.com/anikchand461/lazytune"
              className="mono-label inline-flex items-center gap-2 rounded-md border border-border px-4 py-2.5 text-[12px] text-text transition-colors hover:border-accent hover:text-accent"
            >
              <Github size={15} />
              GitHub
            </a>
          </Section>
        </main>
      </div>
    </>
  );
}

function Section({
  id,
  index,
  children,
}: {
  id: string;
  index: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-24 border-t border-border/60 py-10">
      <h2 className="font-display mb-5 text-[32px] tracking-wide text-text">
        {index}
      </h2>
      {children}
    </section>
  );
}

function MetricGroup({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <div className="mono-label mb-3 text-[10px] text-muted">{title}</div>
      <div className="flex flex-wrap gap-2">
        {items.map((m) => (
          <span
            key={m}
            className="mono-label rounded-md border border-border bg-card px-2.5 py-1.5 text-[11px] text-text/90"
          >
            {m}
          </span>
        ))}
      </div>
    </div>
  );
}

function Table({
  head,
  rows,
}: {
  head: string[];
  rows: React.ReactNode[][];
}) {
  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full min-w-[480px] border-collapse text-left text-[14px]">
        <thead>
          <tr className="border-b border-border bg-card">
            {head.map((h) => (
              <th
                key={h}
                className="mono-label px-4 py-3 text-[10px] font-medium text-muted"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-border/60 last:border-none">
              {r.map((cell, j) => (
                <td key={j} className="px-4 py-3 align-top text-muted">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
