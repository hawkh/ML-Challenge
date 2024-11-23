import TextAnalyzer from '@/components/TextAnalyzer';

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4">
        <h1 className="text-4xl font-bold text-center mb-8">
          AI Text Detector
        </h1>
        <p className="text-center text-gray-600 mb-12">
          Detect whether a text is AI-generated or human-written using our BERT-based model
        </p>
        <TextAnalyzer />
      </div>
    </main>
  );
} 