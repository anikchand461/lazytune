import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0a0a0b",
        surface: "#0f0f11",
        card: "#131316",
        card2: "#17181b",
        border: "#24262b",
        borderHi: "#33353b",
        accent: "#c8ff00",
        accentDim: "#96c200",
        accent2: "#00e5ff",
        danger: "#ff4d4d",
        text: "#f1f1ee",
        muted: "#75787f",
        muted2: "#47494f",
      },
      fontFamily: {
        display: ["var(--font-display)"],
        mono: ["var(--font-mono)"],
        body: ["var(--font-body)"],
      },
      backgroundImage: {
        "grid-lines":
          "linear-gradient(to right, rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.035) 1px, transparent 1px)",
      },
      keyframes: {
        scanline: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        blink: {
          "0%,100%": { opacity: "1" },
          "50%": { opacity: "0.25" },
        },
        floatUp: {
          "0%": { transform: "translateY(6px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
      },
      animation: {
        scanline: "scanline 3.2s linear infinite",
        blink: "blink 1.6s ease-in-out infinite",
        floatUp: "floatUp 0.5s ease forwards",
      },
    },
  },
  plugins: [],
};
export default config;
