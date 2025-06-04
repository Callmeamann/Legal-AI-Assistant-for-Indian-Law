"use client";

import { useState, useRef, useEffect } from 'react';
import { FaPaperPlane, FaSpinner, FaUserCircle, FaRobot, FaRegThumbsUp, FaRegThumbsDown, FaTrash } from 'react-icons/fa';
import { IoMdInformationCircleOutline } from 'react-icons/io';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{
    content: string;
    metadata: {
      source: string;
    };
  }>;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setError('');
    setIsLoading(true);

    // Add user message immediately
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    try {
      const response = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage,
          chat_history: []
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources
      }]);
    } catch (err) {
      setError('Failed to get response. Please try again.');
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Clear chat handler
  const handleClearChat = () => {
    setMessages([]);
    setError('');
  };

  return (
    <div className="flex flex-col h-[800px]">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <div className="mb-4">
              <IoMdInformationCircleOutline className="mx-auto h-12 w-12" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Welcome to Legal AI Assistant</h3>
            <p className="max-w-md mx-auto">
              Ask questions about Indian laws, particularly about the new Bharatiya Nyaya Sanhita (BNS),
              Supreme Court cases, and legal procedures.
            </p>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex w-full ${message.role === 'user' ? 'justify-end' : 'justify-start'} items-start space-x-2`}
          >
            {/* Avatar */}
            {message.role === 'assistant' && (
              <FaRobot className="text-blue-600 mt-1" size={24} />
            )}

            {/* Message bubble */}
            <div
              className={`max-w-[80%] rounded-lg p-4 ${message.role === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-900'
              }`}
            >
              <div className="prose prose-sm sm:prose-base whitespace-pre-wrap max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
              </div>

              {/* Sources Section */}
              {message.sources && message.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-300">
                  <p className="text-sm font-semibold mb-2">Sources:</p>
                  <div className="space-y-2">
                    {message.sources.map((source, idx) => (
                      <div key={idx} className="text-sm">
                        <a
                          href={source.metadata.source}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-500 hover:underline break-all"
                        >
                          {source.metadata.source}
                        </a>
                    </div>
                  ))}
                  </div>
                </div>
              )}

              {/* Feedback icons (non-functional placeholders) */}
              {message.role === 'assistant' && (
                <div className="flex justify-end space-x-3 mt-2 text-gray-400">
                  <FaRegThumbsUp className="cursor-pointer" title="Upvote (coming soon)" />
                  <FaRegThumbsDown className="cursor-pointer" title="Downvote (coming soon)" />
                </div>
              )}
            </div>

            {message.role === 'user' && (
              <FaUserCircle className="text-blue-600 mt-1" size={24} />
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex items-center text-gray-500 space-x-2">
            <FaSpinner className="animate-spin" />
            <span>Generating response...</span>
          </div>
        )}

        {error && (
          <div className="text-red-500 text-center p-2">
            {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="border-t p-4">
        <div className="flex flex-col sm:flex-row sm:space-x-4 space-y-4 sm:space-y-0">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about Indian laws, cases, or legal procedures..."
            className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className={`px-4 py-2 rounded-lg bg-blue-600 text-white flex items-center space-x-2
              ${(isLoading || !input.trim()) ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-700'}`}
          >
            <span>Send</span>
            {isLoading ? (
              <FaSpinner className="animate-spin" />
            ) : (
              <FaPaperPlane />
            )}
          </button>

          {/* Clear Chat Button */}
          <button
            type="button"
            onClick={handleClearChat}
            className="px-4 py-2 rounded-lg bg-gray-200 text-gray-700 hover:bg-gray-300 flex items-center space-x-2"
          >
            <FaTrash />
            <span>Clear</span>
          </button>
        </div>
      </form>
    </div>
  );
}
