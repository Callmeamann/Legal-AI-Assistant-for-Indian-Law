import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        primary: {
          DEFAULT: "#4F46E5", // Indigo-600
          light: "#6366F1", // Indigo-500
          dark: "#4338CA",  // Indigo-700
        },
        secondary: {
          DEFAULT: "#10B981", // Emerald-500
        },
        background: "#F9FAFB", // Cool Gray-50
        card: "#FFFFFF",
        text: {
          primary: "#1F2937", // Cool Gray-800
          secondary: "#6B7280", // Cool Gray-500
        },
      },
    },
  },
  plugins: [],
};
export default config;
