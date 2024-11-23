export default function Header() {
  return (
    <header className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 py-6">
        <h1 className="text-3xl font-bold text-gray-900">AI Text Detector</h1>
        <p className="mt-2 text-gray-600">
          Detect whether a text is AI-generated or human-written using our BERT-based model
        </p>
      </div>
    </header>
  )
} 