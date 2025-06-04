import ChatInterface from "@/components/ChatInterface";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-50">
      {/* Header */}
      <header className="w-full bg-gradient-to-r from-blue-900 to-blue-800 text-white py-6 px-4">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold">Legal AI Assistant</h1>
          <p className="mt-2 text-blue-100">
            Powered by Indian Legal Knowledge Base & RAG Technology
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 w-full max-w-6xl mx-auto p-4 md:p-8">
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <ChatInterface />
        </div>
      </div>

      {/* Footer */}
      <footer className="w-full bg-gray-100 border-t border-gray-200 py-4 px-4">
        <div className="max-w-6xl mx-auto text-center text-gray-600 text-sm">
          <p>Powered by Bharatiya Nyaya Sanhita (BNS) and Supreme Court Cases Database</p>
        </div>
      </footer>
    </main>
  );
}
