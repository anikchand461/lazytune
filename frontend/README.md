# LazyTune — Next.js frontend

A rebuilt, animated Next.js (App Router + TypeScript + Tailwind + Framer
Motion) version of the LazyTune hyperparameter-optimizer UI. Same wizard —
Dataset → Model → Parameters → Results — pointed at the same FastAPI/Flask
backend, with a fresh visual system built around the idea of a grid search
itself: a quiet animated field of "candidate points" in the background, and
a screen → prune → train visualization while a run is in flight.

## Getting started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Pointing at your backend

The app talks to the same two endpoints the original vanilla-JS build used:

- `POST /datasets/upload` (multipart form, field `file`)
- `POST /optimize/` (JSON: `{ model, target, metric, param_grid }`)

By default it targets `https://lazytune.onrender.com`. To point at a local
or different backend, create `.env.local`:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

## Structure

```
app/
  layout.tsx        fonts, metadata, global chrome (loader, background field, toasts)
  page.tsx           the 4-step wizard
  docs/page.tsx       documentation page
components/          Header, Stepper, step cards, form controls, canvas background
lib/
  modelData.ts        model → metric/param metadata (ported from app.js)
  api.ts              fetch calls to the backend
```

## Notes

- Animations: Framer Motion for layout/entrance transitions, a hand-rolled
  `<canvas>` field for the ambient background (no extra dependency).
- Respects `prefers-reduced-motion`.
- No backend code is included here — this folder is the frontend only.
