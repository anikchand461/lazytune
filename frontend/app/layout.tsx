import type { Metadata } from "next";
import { Bebas_Neue, Fira_Code, DM_Sans } from "next/font/google";
import "./globals.css";
import SearchField from "@/components/SearchField";
import Loader from "@/components/Loader";
import { ToastProvider } from "@/components/Toast";

const display = Bebas_Neue({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-display",
});

const mono = Fira_Code({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  variable: "--font-mono",
});

const body = DM_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-body",
});

export const metadata: Metadata = {
  title: "LazyTune — Hyperparameter Optimizer",
  description:
    "Screen, prune, and fully train scikit-learn models fast. Upload a dataset, pick a model, define a parameter grid, and let LazyTune find the best configuration.",
  icons: {
    icon: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${display.variable} ${mono.variable} ${body.variable}`}>
      <body className="font-body bg-bg text-text antialiased">
        <Loader />
        <SearchField />
        <div className="crt-overlay" />
        <ToastProvider>{children}</ToastProvider>
      </body>
    </html>
  );
}
