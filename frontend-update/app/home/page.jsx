"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
    setPreviews(
      selectedFiles.map((file) =>
        file.type.startsWith("image/") ? URL.createObjectURL(file) : "📄 " + file.name
      )
    );
  };

  const handleSubmit = async () => {
    if (files.length === 0) return alert("Please upload at least one file");

    setLoading(true);
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
      const response = await fetch("https://underwriting-at5l.onrender.com/underwrite", { method: "POST", body: formData });
      if (!response.ok) return alert("Error processing files");

      const result = await response.json();
console.log("Underwriting result:", result);

// Store the result in sessionStorage
sessionStorage.setItem("underwritingResult", JSON.stringify(result));

// Then navigate to the analysis page
router.push("/analysis");
    } catch (err) {
      console.error("Upload error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => router.push("/");

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navbar */}
      <header className="border-b bg-white shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center p-4">
          <div className="flex items-center space-x-3">
            <Image src="/underwriting.png" alt="logo" width={50} height={50} />
            <span className="text-xl font-bold text-gray-800">Acquire Underwriting</span>
          </div>
          <nav className="hidden md:flex items-center space-x-4">
            <Link href="/home" className="text-gray-700 hover:text-blue-600 transition">Home</Link>
            <Link href="/about" className="text-gray-700 hover:text-blue-600 transition">About</Link>
            <Button variant="outline" onClick={handleLogout}>Logout</Button>
          </nav>

          {/* Mobile Menu */}
          <Sheet>
            <SheetTrigger className="md:hidden">
              <Button variant="ghost">Menu</Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-64 p-4">
              <ScrollArea className="h-full">
                <Link href="/home" className="block mb-2 text-lg text-gray-700 hover:text-blue-600">Home</Link>
                <Link href="/about" className="block mb-2 text-lg text-gray-700 hover:text-blue-600">About</Link>
                <Button variant="outline" className="w-full mt-4" onClick={handleLogout}>Logout</Button>
              </ScrollArea>
            </SheetContent>
          </Sheet>
        </div>
      </header>

      {/* Hero / Intro Section */}
      <section className="bg-gradient-to-r from-blue-50 to-indigo-50 py-12">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-800 animate-fadeIn">
            Welcome to Acquire Underwriting
          </h1>
          <p className="mt-4 text-gray-600 text-lg animate-fadeIn delay-100">
            Upload your property documents and get detailed AI-powered underwriting analysis instantly.
          </p>
          <p className="mt-2 text-gray-500 text-sm">
            Supported formats: PDF, Excel, CSV, Word, TXT
          </p>
        </div>
      </section>

      {/* File Upload Card */}
      <main className="p-8 max-w-4xl mx-auto -mt-12">
        <Card className="shadow-xl border-none hover:shadow-2xl transition-all duration-300">
          <CardHeader>
            <CardTitle className="text-2xl">Upload Property Documents</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <Label htmlFor="file-upload" className="text-gray-700">Select Files</Label>
              <Input
                id="file-upload"
                type="file"
                multiple
                onChange={handleFileChange}
                className="mt-2 border-gray-300 hover:border-blue-400 transition"
              />
            </div>

            {previews.length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 animate-fadeInUp">
                {previews.map((preview, idx) => (
                  <div
                    key={idx}
                    className="border rounded p-4 bg-white shadow-sm hover:shadow-lg transition-all duration-300 flex items-center justify-center h-24 text-gray-700"
                  >
                    {preview.startsWith("http") ? (
                      <img src={preview} alt={`preview-${idx}`} className="h-full object-contain" />
                    ) : (
                      <span>{preview}</span>
                    )}
                  </div>
                ))}
              </div>
            )}

            <Button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white"
            >
              {loading ? "Processing..." : "Submit for Underwriting"}
            </Button>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
