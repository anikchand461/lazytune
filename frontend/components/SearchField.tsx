"use client";

import { useEffect, useRef } from "react";

/**
 * The site's signature visual: a quiet, continuously-shifting grid of lit
 * cells behind every page. It reads as what LazyTune actually does — many
 * candidate points across a parameter grid, most fading, a few pulsing
 * brighter as they're screened — rather than generic decorative particles.
 */
export default function SearchField() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    let raf = 0;
    let W = 0;
    let H = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    const SPACING = window.innerWidth < 640 ? 46 : 58;

    function resize() {
      W = window.innerWidth;
      H = window.innerHeight;
      canvas!.width = W * dpr;
      canvas!.height = H * dpr;
      canvas!.style.width = W + "px";
      canvas!.style.height = H + "px";
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);

    let t = 0;
    // a handful of cells get a slower "scoring" pulse — like trials being ranked
    const scoring = Array.from({ length: 5 }, () => ({
      col: Math.random(),
      row: Math.random(),
      phase: Math.random() * Math.PI * 2,
    }));

    function frame() {
      ctx!.clearRect(0, 0, W, H);
      const cols = Math.ceil(W / SPACING) + 1;
      const rows = Math.ceil(H / SPACING) + 1;

      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const x = i * SPACING;
          const y = j * SPACING;
          const n =
            (Math.sin(i * 0.35 + t * 0.6) +
              Math.sin(j * 0.4 - t * 0.5) +
              Math.sin((i + j) * 0.22 + t * 0.35)) /
            3; // -1..1
          const b = (n + 1) / 2; // 0..1
          if (b < 0.62) continue;
          const alpha = (b - 0.62) / 0.38;
          ctx!.fillStyle = `rgba(200,255,0,${(alpha * 0.4).toFixed(3)})`;
          const s = 1.4 + alpha * 1.6;
          ctx!.fillRect(x - s / 2, y - s / 2, s, s);
        }
      }

      // slow-pulsing "best trial" markers with a faint ring
      scoring.forEach((s) => {
        const x = s.col * W;
        const y = s.row * H;
        const p = (Math.sin(t * 0.5 + s.phase) + 1) / 2;
        ctx!.beginPath();
        ctx!.arc(x, y, 3 + p * 2, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(0,229,255,${(0.12 + p * 0.22).toFixed(3)})`;
        ctx!.fill();
        ctx!.beginPath();
        ctx!.arc(x, y, 10 + p * 10, 0, Math.PI * 2);
        ctx!.strokeStyle = `rgba(0,229,255,${(0.05 + p * 0.08).toFixed(3)})`;
        ctx!.lineWidth = 1;
        ctx!.stroke();
      });

      t += 0.008;
      if (!reduceMotion) raf = requestAnimationFrame(frame);
    }

    frame();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={ref}
      aria-hidden
      className="pointer-events-none fixed inset-0 z-0 opacity-90"
    />
  );
}
