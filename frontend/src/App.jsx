import { useState } from "react";
import UploadForm from "./components/UploadForm";
import AnalysisPage from "./components/AnalysisPage";

export default function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [tab, setTab] = useState("new");

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      {/* Navbar */}
      <nav className="flex gap-4 mb-6">
        <button
          onClick={() => setTab("new")}
          className={`px-4 py-2 rounded-lg ${tab === "new" ? "bg-blue-600 text-white" : "bg-white shadow"}`}
        >
          New Analysis
        </button>
        <button
          onClick={() => setTab("analysis")}
          disabled={!analysisData}
          className={`px-4 py-2 rounded-lg ${tab === "analysis" ? "bg-blue-600 text-white" : "bg-white shadow"} disabled:opacity-50`}
        >
          Analysis
        </button>
      </nav>

      {tab === "new" && <UploadForm onAnalysisComplete={setAnalysisData} />}
      {tab === "analysis" && <AnalysisPage data={analysisData} />}
    </div>
  );
}
