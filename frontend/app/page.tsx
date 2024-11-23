import TextAnalyzer from '@/components/TextAnalyzer'
import Header from '@/components/Header'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      <Header />
      <div className="max-w-7xl mx-auto px-4 py-12">
        <TextAnalyzer />
      </div>
    </main>
  )
} 