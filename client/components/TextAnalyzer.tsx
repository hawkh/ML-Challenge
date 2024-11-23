'use client';

import { useState } from 'react';

export default function TextAnalyzer() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<null | {
    prediction: string;
    confidence: number;
  }>(null);
  const [loading, setLoading] = useState(false);

  const analyzeText = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <form onSubmit={analyzeText} className="space-y-4">
        <textarea
          className="w-full h-48 p-4 border rounded-lg focus:ring-2 focus:ring-blue-500"
          placeholder="Enter text to analyze..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button
          type="submit"
          disabled={loading || !text}
          className="w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? 'Analyzing...' : 'Analyze Text'}
        </button>
      </form>

      {result && (
        <div className="mt-8 p-4 border rounded-lg">
          <h3 className="text-xl font-bold mb-2">Result:</h3>
          <p className="text-lg">
            This text appears to be{' '}
            <span className={`font-bold ${
              result.prediction === 'AI-generated' ? 'text-red-500' : 'text-green-500'
            }`}>
              {result.prediction}
            </span>
          </p>
          <p className="mt-2">
            Confidence: {(result.confidence * 100).toFixed(2)}%
          </p>
        </div>
      )}
    </div>
  );
} 