/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Environment variables that should be available on the client side
  // Make sure your FastAPI backend URL is set in your .env.local file
  // e.g., NEXT_PUBLIC_API_URL=http://localhost:8000
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
};

export default nextConfig;
