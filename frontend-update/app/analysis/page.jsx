"use client";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from "recharts";

export default function AnalysisPage() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const stored = sessionStorage.getItem("underwritingResult");
    if (!stored) return;
    setData(JSON.parse(stored));
  }, []);

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-600 text-lg">No analysis data found. Please upload files first.</p>
      </div>
    );
  }


  const {
    quick_summary = {},
    metrics = {},
    ai_summary = "",
    ai_analysis = {},
    t12_summary = {},
  } = data;

  const financialMetricsData = [
    { name: "Cap Rate", value: metrics.cap_rate ?? 0 },
    { name: "DSCR", value: metrics.dscr ?? 0 },
    { name: "CoC Return", value: metrics.coc_return ?? 0 },
    { name: "Break-even Occupancy", value: metrics.break_even_occupancy ?? 0 },
  ];

  const expenseData = [
    { name: "Gross Potential Rent", value: t12_summary.gross_potential_rent ?? 0 },
    { name: "Net Operating Income", value: t12_summary.net_operating_income ?? 0 },
    { name: "Operating Expenses", value: t12_summary.operating_expenses ?? 0 },
  ];

  const COLORS = ["#4f46e5", "#3b82f6", "#06b6d4", "#f97316"];

  console.log("📊 Investment Recommendation:", quick_summary["Investment Recommendation"]);
  console.log("\n🌟 Key Investment Highlights:");
  quick_summary["Key Investment Highlights"].forEach((highlight, index) => {
    console.log(`${index + 1}. ${highlight}`);
  });
  
  console.log("\n⚠️ Risk Considerations:");
  quick_summary["Risk Considerations"].forEach((risk, index) => {
    console.log(`${index + 1}. ${risk}`);
  });

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-3xl font-bold text-gray-800 mb-6 animate-fadeIn">
        Underwriting Analysis for {quick_summary.property || "N/A"}
      </h1>

      <ScrollArea className="space-y-6 max-w-6xl mx-auto">
        {/* Property Details */}
        <Card className="animate-fadeInUp">
          <CardHeader>
            <CardTitle>Property Details</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <p><strong>Address:</strong> {quick_summary.address || "N/A"}</p>
            <p><strong>Year Built:</strong> {quick_summary.year_built || "N/A"}</p>
            <p><strong>Total SqFt:</strong> {quick_summary.sqft || "N/A"}</p>
            <p><strong>Price per SqFt:</strong> {metrics.price_per_sqft ?? "N/A"}</p>
            <p><strong>Price per Unit:</strong> {metrics.price_per_unit ?? "N/A"}</p>
          </CardContent>
        </Card>

        {/* Financial Metrics */}
        <Card className="animate-fadeInUp">
          <CardHeader>
            <CardTitle>Financial Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={financialMetricsData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(value) => (typeof value === "number" ? value.toFixed(2) : value)} />
                <Bar dataKey="value" fill="#4f46e5" animationDuration={1500} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Operating Summary */}
        <Card className="animate-fadeInUp">
          <CardHeader>
            <CardTitle>Operating Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={expenseData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label
                >
                  {expenseData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Legend />
                <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* AI Summary */}
        <Card className="animate-fadeInUp">
          <CardHeader>
            <CardTitle>AI Underwriting Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p>{ai_summary || "No AI summary available."}</p>
          </CardContent>
        </Card>

        

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


      </ScrollArea>
    </div>
  );
}



// "use client";
// import { useEffect, useState } from "react";
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
// import { ScrollArea } from "@/components/ui/scroll-area";
// import {
//   BarChart,
//   Bar,
//   XAxis,
//   YAxis,
//   Tooltip,
//   ResponsiveContainer,
//   PieChart,
//   Pie,
//   Cell,
//   Legend,
// } from "recharts";

// export default function AnalysisPage() {
//   const [data, setData] = useState(null);

//   useEffect(() => {
//     const stored = sessionStorage.getItem("underwritingResult");
//     if (!stored) return;
//     setData(JSON.parse(stored));
//   }, []);

//   if (!data) {
//     return (
//       <div className="min-h-screen flex items-center justify-center">
//         <p className="text-gray-600 text-lg">
//           No analysis data found. Please upload files first.
//         </p>
//       </div>
//     );
//   }

//   const {
//     quick_summary = {},
//     metrics = {},
//     ai_summary = "",
//     ai_analysis = {},
//     t12_summary = {},
//     irr_summary = {},
//     sensitivity_analysis = {},
//   } = data;

//   // Financial metrics chart
//   const financialMetricsData = [
//     { name: "Cap Rate", value: metrics.cap_rate ?? 0 },
//     { name: "DSCR", value: metrics.dscr ?? 0 },
//     { name: "CoC Return", value: metrics.coc_return ?? 0 },
//     { name: "Break-even Occupancy", value: metrics.break_even_occupancy ?? 0 },
//   ];

//   // Operating expenses chart
//   const expenseData = [
//     {
//       name: "Gross Potential Rent",
//       value: t12_summary.gross_potential_rent ?? 0,
//     },
//     { name: "Net Operating Income", value: t12_summary.net_operating_income ?? 0 },
//     { name: "Operating Expenses", value: t12_summary.operating_expenses ?? 0 },
//   ];

//   const COLORS = ["#4f46e5", "#3b82f6", "#06b6d4", "#f97316"];

//   // Debugging logs
//   console.log("📊 Investment Recommendation:", quick_summary["Investment Recommendation"]);
//   if (Array.isArray(quick_summary["Key Investment Highlights"])) {
//     console.log("\n🌟 Key Investment Highlights:");
//     quick_summary["Key Investment Highlights"].forEach((highlight, index) => {
//       console.log(`${index + 1}. ${highlight}`);
//     });
//   }
//   if (Array.isArray(quick_summary["Risk Considerations"])) {
//     console.log("\n⚠️ Risk Considerations:");
//     quick_summary["Risk Considerations"].forEach((risk, index) => {
//       console.log(`${index + 1}. ${risk}`);
//     });
//   }

//   return (
//     <div className="min-h-screen bg-gray-50 p-8">
//       <h1 className="text-3xl font-bold text-gray-800 mb-6 animate-fadeIn">
//         Underwriting Analysis for {quick_summary.property || "N/A"}
//       </h1>

//       <ScrollArea className="space-y-6 max-w-6xl mx-auto">
//         {/* Property Details */}
//         <Card className="animate-fadeInUp">
//           <CardHeader>
//             <CardTitle>Property Details</CardTitle>
//           </CardHeader>
//           <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4">
//             <p>
//               <strong>Address:</strong> {quick_summary.address || "N/A"}
//             </p>
//             <p>
//               <strong>Year Built:</strong> {quick_summary.year_built || "N/A"}
//             </p>
//             <p>
//               <strong>Total SqFt:</strong> {quick_summary.sqft || "N/A"}
//             </p>
//             <p>
//               <strong>Price per SqFt:</strong> {metrics.price_per_sqft ?? "N/A"}
//             </p>
//             <p>
//               <strong>Price per Unit:</strong> {metrics.price_per_unit ?? "N/A"}
//             </p>
//           </CardContent>
//         </Card>

//         {/* Financial Metrics */}
//         <Card className="animate-fadeInUp">
//           <CardHeader>
//             <CardTitle>Financial Metrics</CardTitle>
//           </CardHeader>
//           <CardContent>
//             <ResponsiveContainer width="100%" height={250}>
//               <BarChart
//                 data={financialMetricsData}
//                 margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
//               >
//                 <XAxis dataKey="name" />
//                 <YAxis />
//                 <Tooltip
//                   formatter={(value) =>
//                     typeof value === "number" ? value.toFixed(2) : value
//                   }
//                 />
//                 <Bar dataKey="value" fill="#4f46e5" animationDuration={1500} />
//               </BarChart>
//             </ResponsiveContainer>
//           </CardContent>
//         </Card>

//         {/* Operating Summary */}
//         <Card className="animate-fadeInUp">
//           <CardHeader>
//             <CardTitle>Operating Summary</CardTitle>
//           </CardHeader>
//           <CardContent>
//             <ResponsiveContainer width="100%" height={250}>
//               <PieChart>
//                 <Pie
//                   data={expenseData}
//                   dataKey="value"
//                   nameKey="name"
//                   cx="50%"
//                   cy="50%"
//                   outerRadius={80}
//                   label
//                 >
//                   {expenseData.map((entry, index) => (
//                     <Cell
//                       key={`cell-${index}`}
//                       fill={COLORS[index % COLORS.length]}
//                     />
//                   ))}
//                 </Pie>
//                 <Legend />
//                 <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
//               </PieChart>
//             </ResponsiveContainer>
//           </CardContent>
//         </Card>

//         {/* IRR Analysis */}
//         <Card className="animate-fadeInUp">
//           <CardHeader>
//             <CardTitle>IRR Analysis</CardTitle>
//           </CardHeader>
//           <CardContent>
//             <ul className="list-disc list-inside space-y-1">
//               <li>
//                 <strong>5-Year Levered IRR:</strong>{" "}
//                 {irr_summary?.irr_5y_levered ?? "N/A"}
//               </li>
//               <li>
//                 <strong>5-Year Unlevered IRR:</strong>{" "}
//                 {irr_summary?.irr_5y_unlevered ?? "N/A"}
//               </li>
//               <li>
//                 <strong>10-Year Levered IRR:</strong>{" "}
//                 {irr_summary?.irr_10y_levered ?? "N/A"}
//               </li>
//               <li>
//                 <strong>10-Year Unlevered IRR:</strong>{" "}
//                 {irr_summary?.irr_10y_unlevered ?? "N/A"}
//               </li>
//             </ul>
//           </CardContent>
//         </Card>

//         {/* Sensitivity Analysis */}
//         <Card className="animate-fadeInUp">
//           <CardHeader>
//             <CardTitle>Sensitivity Analysis</CardTitle>
//           </CardHeader>
//           <CardContent className="space-y-4">
//             {/* Cap Rate Variation */}
//             <div>
//               <strong>Cap Rate Sensitivity:</strong>
//               {Array.isArray(sensitivity_analysis?.cap_rate) ? (
//                 <ul className="list-disc list-inside mt-1 space-y-1">
//                   {sensitivity_analysis.cap_rate.map((item, index) => (
//                     <li key={index}>{item}</li>
//                   ))}
//                 </ul>
//               ) : (
//                 <p>N/A</p>
//               )}
//             </div>

//             {/* Loan Term Variation */}
//             <div>
//               <strong>Loan Term Sensitivity:</strong>
//               {Array.isArray(sensitivity_analysis?.loan_term) ? (
//                 <ul className="list-disc list-inside mt-1 space-y-1">
//                   {sensitivity_analysis.loan_term.map((item, index) => (
//                     <li key={index}>{item}</li>
//                   ))}
//                 </ul>
//               ) : (
//                 <p>N/A</p>
//               )}
//             </div>
//           </CardContent>
//         </Card>

//         {/* AI Summary */}
//         <Card className="animate-fadeInUp">
//           <CardHeader>
//             <CardTitle>AI Underwriting Summary</CardTitle>
//           </CardHeader>
//           <CardContent className="space-y-2">
//             <p>{ai_summary || "No AI summary available."}</p>
//           </CardContent>
//         </Card>

//         {/* AI Analysis */}
//         <Card className="animate-fadeInUp">
//           <CardHeader>
//             <CardTitle>AI Analysis & Recommendations</CardTitle>
//           </CardHeader>

//           <CardContent className="space-y-4">
//             {/* Investment Recommendation */}
//             <p>
//               <strong>Investment Recommendation:</strong>{" "}
//               {quick_summary["Investment Recommendation"] || "N/A"}
//             </p>

//             {/* Key Highlights */}
//             <div>
//               <strong>Key Highlights:</strong>
//               {Array.isArray(quick_summary["Key Investment Highlights"]) ? (
//                 <ul className="list-disc list-inside mt-1 space-y-1">
//                   {quick_summary["Key Investment Highlights"].map(
//                     (highlight, index) => (
//                       <li key={index}>{highlight}</li>
//                     )
//                   )}
//                 </ul>
//               ) : (
//                 <p>N/A</p>
//               )}
//             </div>

//             {/* Risk Considerations */}
//             <div>
//               <strong>Risk Considerations:</strong>
//               {Array.isArray(quick_summary["Risk Considerations"]) ? (
//                 <ul className="list-disc list-inside mt-1 space-y-1">
//                   {quick_summary["Risk Considerations"].map((risk, index) => (
//                     <li key={index}>{risk}</li>
//                   ))}
//                 </ul>
//               ) : (
//                 <p>N/A</p>
//               )}
//             </div>
//           </CardContent>
//         </Card>
//       </ScrollArea>
//     </div>
//   );
// }
