import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Legal AI Assistant - Indian Law Expert",
  description: "AI-powered legal assistant specializing in Indian laws, including BNS, Supreme Court cases, and legal procedures.",
  keywords: "legal ai, indian law, bharatiya nyaya sanhita, supreme court, legal assistant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
