

"use client";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useRef } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  Line,
  ComposedChart,
} from "recharts";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Download,
  TrendingUp,
  CheckCircle,
  Building,
  DollarSign,
  AlertTriangle,
} from "lucide-react";

export default function AnalysisPage() {
  const memoRef = useRef();
  const router = useRouter();
  const [data, setData] = useState(null);

  useEffect(() => {
    const stored = sessionStorage.getItem("underwritingResult");
    if (!stored) return;
    setData(JSON.parse(stored));
  }, []);

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-600 text-lg">
          No analysis data found. Please upload files first.
        </p>
      </div>
    );
  }

  const {
    quick_summary = {},
    metrics = {},
    ai_summary = "",
    t12_summary = {},
  } = data;

  const financialMetricsData = [
    { name: "Cap Rate", value: metrics.cap_rate ?? 0 },
    { name: "DSCR", value: metrics.dscr ?? 0 },
    { name: "CoC Return", value: metrics.coc_return ?? 0 },
    { name: "Break-even Occupancy", value: metrics.break_even_occupancy ?? 0 },
  ];

  const expenseData = [
    {
      name: "Gross Potential Rent",
      value: t12_summary.gross_potential_rent ?? 0,
    },
    { name: "Net Operating Income", value: t12_summary.net_operating_income ?? 0 },
    { name: "Operating Expenses", value: t12_summary.operating_expenses ?? 0 },
  ];

  const rentGapData = [
    { name: "Current", value: 19.2, color: "#f87171" },
    { name: "Market", value: 20.78, color: "#34d399" },
  ];

  const fiveYearProjection = [
    { year: "Year 1", revenue: 100000, expenses: 60000, noi: 40000 },
    { year: "Year 2", revenue: 105000, expenses: 62000, noi: 43000 },
    { year: "Year 3", revenue: 110250, expenses: 64000, noi: 46250 },
    { year: "Year 4", revenue: 115763, expenses: 66000, noi: 49763 },
    { year: "Year 5", revenue: 121551, expenses: 68000, noi: 53551 },
  ];

  const COLORS = ["#6366f1", "#60a5fa", "#22d3ee", "#fbbf24"];

  const getRecommendationColor = (recommendation) => {
    if (!recommendation) return "bg-gray-500";
    const rec = recommendation.toLowerCase();
    if (rec.includes("buy") || rec.includes("strong buy")) return "bg-green-400";
    if (rec.includes("hold")) return "bg-yellow-400";
    if (rec.includes("pass") || rec.includes("avoid")) return "bg-red-400";
    return "bg-blue-400";
  };

  const handleDownloadPDF = async () => {
    if (!memoRef.current) return;

    const html2pdf = (await import("html2pdf.js")).default;

    const element = memoRef.current;
    html2pdf().from(element).save("analysis.pdf");
  };

  

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <Button
        variant="default"
        className="absolute top-4 right-4 z-50"
        onClick={() => router.push("/home")}
      >
        Back to Home
      </Button>

      <h1 className="text-3xl font-bold text-gray-800 mb-6">
        Underwriting Analysis for {quick_summary.property || "N/A"}
      </h1>

      <ScrollArea className="space-y-6 max-w-6xl mx-auto">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Purchase Price</p>
                  <p className="text-2xl font-bold text-gray-900">
                    ${(metrics.purchase_price || 0).toLocaleString()}
                  </p>
                </div>
                <DollarSign className="h-8 w-8 text-blue-400" />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Investment Recommendation</p>
                  <Badge
                    className={`${getRecommendationColor(
                      quick_summary["Investment Recommendation"]
                    )} text-white`}
                  >
                    {quick_summary["Investment Recommendation"] || "PASS"}
                  </Badge>
                </div>
                <TrendingUp className="h-8 w-8 text-green-400" />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Analysis Confidence</p>
                  <p className="text-2xl font-bold text-gray-900">55%</p>
                  <Progress value={55} className="mt-2" />
                </div>
                <CheckCircle className="h-8 w-8 text-purple-400" />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Net Operating Income</p>
                  <p className="text-2xl font-bold text-gray-900">
                    ${(t12_summary.net_operating_income || 0).toLocaleString()}
                  </p>
                </div>
                <Building className="h-8 w-8 text-orange-400" />
              </div>
            </CardContent>
          </Card>
        </div>

      {/* Financial Metrics & Operating Summary */}
<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
  {/* Financial Metrics - Bar Chart */}
  <Card className="bg-white/70 backdrop-blur-sm shadow-md border border-gray-100">
    <CardHeader>
      <CardTitle className="text-gray-800">Financial Metrics</CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={financialMetricsData}>
          <XAxis dataKey="name" stroke="#9CA3AF" /> {/* lighter gray axis */}
          <YAxis stroke="#9CA3AF" />
          <Tooltip
            contentStyle={{ backgroundColor: "rgba(255,255,255,0.9)", border: "1px solid #E5E7EB", borderRadius: "8px" }}
          />
          <Bar
            dataKey="value"
            fill="rgba(59,130,246,0.6)"  // soft translucent blue
            radius={[6, 6, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>

  {/* Operating Summary - Pie Chart */}
  <Card className="bg-white/70 backdrop-blur-sm shadow-md border border-gray-100">
    <CardHeader>
      <CardTitle className="text-gray-800">Operating Summary</CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={expenseData}
            dataKey="value"
            nameKey="name"
            outerRadius={90}
            label
          >
            {expenseData.map((_, i) => (
              <Cell
                key={i}
                fill={[
                  "rgba(59,130,246,0.6)",   // soft blue
                  "rgba(16,185,129,0.6)",  // soft teal
                  "rgba(139,92,246,0.6)",  // soft purple
                  "rgba(245,158,11,0.6)",  // soft amber
                  "rgba(107,114,128,0.5)", // muted gray
                ][i % 5]}
              />
            ))}
          </Pie>
          <Legend />
          <Tooltip
            contentStyle={{ backgroundColor: "rgba(255,255,255,0.9)", border: "1px solid #E5E7EB", borderRadius: "8px" }}
          />
        </PieChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
</div>

{/* Rent Gap & 5-Year Projection */}
<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
  {/* Rent Gap */}
  <Card className="shadow-lg rounded-2xl border border-gray-200 bg-white/70 backdrop-blur">
    <CardHeader>
      <CardTitle className="text-gray-800">Rent Gap Analysis</CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={rentGapData}>
          <XAxis dataKey="name" stroke="#6b7280" />
          <YAxis stroke="#6b7280" />
          <Tooltip contentStyle={{ backgroundColor: "rgba(255,255,255,0.9)", borderRadius: "12px" }} />
          <Bar
            dataKey="value"
            fill="rgba(59, 130, 246, 0.2)"   // light blue with transparency
            stroke="#3b82f6"
            strokeWidth={2}
            radius={[6, 6, 0, 0]}
          >
            {rentGapData.map((d, i) => (
              <Cell key={i} fill={d.color + "33"} stroke={d.color} /> // adds transparency per-bar
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>

  {/* 5-Year Projection */}
  <Card className="shadow-lg rounded-2xl border border-gray-200 bg-white/70 backdrop-blur">
    <CardHeader>
      <CardTitle className="text-gray-800">5-Year Financial Projection</CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={fiveYearProjection}>
          <XAxis dataKey="year" stroke="#6b7280" />
          <YAxis stroke="#6b7280" />
          <Tooltip contentStyle={{ backgroundColor: "rgba(255,255,255,0.9)", borderRadius: "12px" }} />
          <Legend />
          <Bar
            dataKey="revenue"
            fill="rgba(59, 130, 246, 0.2)"   // soft blue transparent
            stroke="#3b82f6"
            strokeWidth={2}
            radius={[6, 6, 0, 0]}
            name="Revenue"
          />
          <Bar
            dataKey="expenses"
            fill="rgba(239, 68, 68, 0.2)"   // soft red transparent
            stroke="#ef4444"
            strokeWidth={2}
            radius={[6, 6, 0, 0]}
            name="Expenses"
          />
          <Line
            dataKey="noi"
            stroke="#10b981"
            strokeWidth={2.5}
            dot={{ r: 4, fill: "#10b981" }}
            name="NOI"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
</div>



       {/* AI Summary */}
<Card className="animate-fadeInUp">
  <CardHeader>
    <CardTitle>AI Underwriting Summary</CardTitle>
  </CardHeader>
  <CardContent className="space-y-2">
    <p>{ai_summary || "No AI summary available."}</p>
  </CardContent>
</Card>

{/* Enhanced Key Insights and Risk Factors */}
<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
  {/* Key Highlights */}
  <Card className="animate-fadeInUp">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <CheckCircle className="w-5 h-5 text-green-500" />
        Key Investment Highlights
      </CardTitle>
    </CardHeader>
    <CardContent>
      {Array.isArray(quick_summary["Key Investment Highlights"]) ? (
        <ul className="space-y-3">
          {quick_summary["Key Investment Highlights"].map((highlight, index) => (
            <li key={index} className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
              <span className="text-sm text-gray-700">{highlight}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-gray-500 italic">Client-only mode: limited insights</p>
      )}
    </CardContent>
  </Card>

  {/* Risk Considerations */}
  <Card className="animate-fadeInUp">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <AlertTriangle className="w-5 h-5 text-red-500" />
        Risk Considerations
      </CardTitle>
    </CardHeader>
    <CardContent>
      {Array.isArray(quick_summary["Risk Considerations"]) ? (
        <ul className="space-y-3">
          {quick_summary["Risk Considerations"].map((risk, index) => (
            <li key={index} className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
              <span className="text-sm text-gray-700">{risk}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-gray-500 italic">Client-only mode: limited risks</p>
      )}
    </CardContent>
  </Card>
</div>

{/* AI Analysis */}
<Card className="animate-fadeInUp">
  <CardHeader>
    <CardTitle>AI Analysis & Recommendations</CardTitle>
  </CardHeader>
  <CardContent className="space-y-4">
    {/* Investment Recommendation */}
    <p>
      <strong>Investment Recommendation:</strong>{" "}
      {quick_summary["Investment Recommendation"] || "N/A"}
    </p>

    {/* Key Highlights */}
    <div>
      <strong>Key Highlights:</strong>
      {Array.isArray(quick_summary["Key Investment Highlights"]) ? (
        <ul className="list-disc list-inside mt-1 space-y-1">
          {quick_summary["Key Investment Highlights"].map((highlight, index) => (
            <li key={index}>{highlight}</li>
          ))}
        </ul>
      ) : (
        <p>N/A</p>
      )}
    </div>

    {/* Risk Considerations */}
    <div>
      <strong>Risk Considerations:</strong>
      {Array.isArray(quick_summary["Risk Considerations"]) ? (
        <ul className="list-disc list-inside mt-1 space-y-1">
          {quick_summary["Risk Considerations"].map((risk, index) => (
            <li key={index}>{risk}</li>
          ))}
        </ul>
      ) : (
        <p>N/A</p>
      )}
    </div>
  </CardContent>
</Card>

{/* New Deal Memo */}
<Card ref={memoRef} className="animate-fadeInUp">
  <CardHeader>
    <CardTitle className="flex items-center justify-between">
      Deal Memo - Investment Analysis Summary
      <Button
        onClick={handleDownloadPDF}
        variant="outline"
        size="sm"
        className="flex items-center gap-2"
      >
        <Download className="w-4 h-4" /> Download PDF
      </Button>
    </CardTitle>
  </CardHeader>

  <CardContent className="space-y-6">
    {/* Executive Summary */}
    <div>
      <h3 className="text-lg font-semibold mb-3">Executive Summary</h3>
      <div className="bg-gray-50 p-4 rounded-lg">
        <p className="text-sm text-gray-700">
          {ai_summary ||
            "Comprehensive investment analysis for this property reveals key financial metrics and market positioning factors that influence the investment decision."}
        </p>
      </div>
    </div>

    {/* Property Details Grid */}
    <div>
      <h3 className="text-lg font-semibold mb-3">Property Details</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm font-medium text-gray-600">Property Name</p>
          <p className="text-lg font-semibold">
            {quick_summary.property || "Property Analysis"}
          </p>
        </div>
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm font-medium text-gray-600">Address</p>
          <p className="text-lg font-semibold">
            {quick_summary.address || "N/A"}
          </p>
        </div>
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm font-medium text-gray-600">Type</p>
          <p className="text-lg font-semibold">Mixed Use</p>
        </div>
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm font-medium text-gray-600">Total Units</p>
          <p className="text-lg font-semibold">{quick_summary.units || 0}</p>
        </div>
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm font-medium text-gray-600">Square Footage</p>
          <p className="text-lg font-semibold">
            {(quick_summary.sqft || 0).toLocaleString()}
          </p>
        </div>
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm font-medium text-gray-600">Year Built</p>
          <p className="text-lg font-semibold">
            {quick_summary.year_built || "2020"}
          </p>
        </div>
      </div>
    </div>

    {/* Financial Summary */}
    <div>
      <h3 className="text-lg font-semibold mb-3">Financial Summary</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg text-center">
          <p className="text-sm font-medium text-blue-600">Revenue</p>
          <p className="text-2xl font-bold text-blue-900">
            ${(
              t12_summary.gross_potential_rent || 0
            ).toLocaleString()}
          </p>
        </div>
        <div className="bg-red-50 border border-red-200 p-4 rounded-lg text-center">
          <p className="text-sm font-medium text-red-600">Expenses</p>
          <p className="text-2xl font-bold text-red-900">
            ${(t12_summary.operating_expenses || 0).toLocaleString()}
          </p>
        </div>
        <div className="bg-green-50 border border-green-200 p-4 rounded-lg text-center">
          <p className="text-sm font-medium text-green-600">
            Net Operating Income
          </p>
          <p className="text-2xl font-bold text-green-900">
            ${(t12_summary.net_operating_income || 0).toLocaleString()}
          </p>
        </div>
        <div className="bg-purple-50 border border-purple-200 p-4 rounded-lg text-center">
          <p className="text-sm font-medium text-purple-600">Cap Rate</p>
          <p className="text-2xl font-bold text-purple-900">
            {(metrics.cap_rate || 0).toFixed(2)}%
          </p>
        </div>
      </div>
    </div>

    {/* Footer */}
    <div className="border-t pt-4 text-center text-sm text-gray-500">
      <p>
        Generated on{" "}
        {new Date().toLocaleDateString("en-US", {
          year: "numeric",
          month: "long",
          day: "numeric",
        })}
      </p>
      <p className="font-semibold">RE Underwriting Platform</p>
    </div>
  </CardContent>
</Card>
</ScrollArea> </div> ); }
