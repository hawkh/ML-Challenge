'use client';

import { useState } from 'react';
import LoadingSpinner from './LoadingSpinner';

interface AnalysisResult {
  prediction: string;
  confidence: number;
}

export default function TextAnalyzer() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeText = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed. Please try again.');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-2xl mx-auto">
      <form onSubmit={analyzeText} className="space-y-4">
        <div>
          <label htmlFor="text" className="block text-sm font-medium text-gray-700 mb-2">
            Enter your text
          </label>
          <textarea
            id="text"
            className="w-full h-48 p-4 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="Paste your text here to analyze..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>

        <button
          type="submit"
          disabled={loading || !text}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 
                   disabled:opacity-50 disabled:cursor-not-allowed transition-colors
                   flex items-center justify-center"
        >
          {loading ? (
            <>
              <LoadingSpinner />
              <span className="ml-2">Analyzing...</span>
            </>
          ) : (
            'Analyze Text'
          )}
        </button>
      </form>

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      {result && (
        <div className="mt-8 p-4 bg-gray-50 border rounded-lg">
          <h3 className="text-xl font-bold mb-2">Analysis Result</h3>
          <div className="space-y-2">
            <p className="text-lg">
              This text appears to be{' '}
              <span className={`font-bold ${
                result.prediction === 'AI-generated' ? 'text-red-600' : 'text-green-600'
              }`}>
                {result.prediction}
              </span>
            </p>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className={`h-2.5 rounded-full ${
                  result.prediction === 'AI-generated' ? 'bg-red-600' : 'bg-green-600'
                }`}
                style={{ width: `${result.confidence * 100}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-600">
              Confidence: {(result.confidence * 100).toFixed(2)}%
            </p>
          </div>
        </div>
      )}
    </div>
  )
} 