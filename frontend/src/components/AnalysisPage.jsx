// "use client";

// import { useState } from "react";
// import { CSVLink } from "react-csv";
// import {
//   Box,
//   Card,
//   CardContent,
//   CardHeader,
//   Typography,
//   Grid,
//   Paper,
//   TableContainer,
//   Table,
//   TableHead,
//   TableRow,
//   TableCell,
//   TableBody,
// } from "@mui/material";
// import {
//   LineChart,
//   Line,
//   XAxis,
//   YAxis,
//   CartesianGrid,
//   Tooltip,
//   ResponsiveContainer,
//   BarChart,
//   Bar,
// } from "recharts";

// export default function AnalysisPage({ data }) {
//   if (!data)
//     return (
//       <Typography
//         variant="body1"
//         color="text.secondary"
//         align="center"
//         sx={{ mt: 10, fontSize: "1.2rem" }}
//       >
//         No analysis yet.
//       </Typography>
//     );

//   const { narrative_fields, metrics, t12_summary, tables, ai_summary } = data;

//   // CSV export data
//   const csvData = [
//     ["Property Name", narrative_fields.property_name],
//     ["Address", narrative_fields.property_address],
//     ["Type", narrative_fields.property_type],
//     ["Year Built", narrative_fields.year_built],
//     ["Total Units / Suites", narrative_fields.total_units_or_suites],
//     ["SqFt", narrative_fields.total_building_sqft],
//     ["Amenities", narrative_fields.amenities || "N/A"],
//     [],
//     ["Financial Metrics"],
//     ["Purchase Price", "$5,000,000"],
//     ["Cap Rate", metrics.cap_rate ? (metrics.cap_rate * 100).toFixed(1) + "%" : "N/A"],
//     ["DSCR", metrics.dscr?.toFixed(2) || "N/A"],
//     ["CoC Return", metrics.coc_return ? (metrics.coc_return * 100).toFixed(1) + "%" : "N/A"],
//     ["IRR (5yr)", metrics.irr_5yr ? metrics.irr_5yr.toFixed(1) + "%" : "N/A"],
//     ["Rent Gap %", metrics.rent_gap_pct ? (metrics.rent_gap_pct).toFixed(1) + "%" : "N/A"],
//     ["Price per SqFt", metrics.price_per_sqft ? `$${metrics.price_per_sqft}` : "N/A"],
//     ["Price per Unit", metrics.price_per_unit ? `$${metrics.price_per_unit}` : "N/A"],
//     ["Break-even Occupancy", metrics.break_even_occupancy ? metrics.break_even_occupancy.toFixed(1) + "%" : "N/A"],
//     [],
//     ["T12 Summary"],
//     ["Gross Potential Rent", t12_summary.gross_potential_rent],
//     ["Vacancy", t12_summary.vacancy],
//     ["Effective Gross Income", t12_summary.effective_gross_income],
//     ["Operating Expenses", t12_summary.operating_expenses],
//     ["Net Operating Income", t12_summary.net_operating_income],
//   ];

//   // Chart data for T12
//   const t12ChartData = [
//     { name: "Gross Rent", value: t12_summary.gross_potential_rent },
//     { name: "Vacancy", value: t12_summary.vacancy },
//     { name: "Operating Expenses", value: t12_summary.operating_expenses },
//     { name: "Net Income", value: t12_summary.net_operating_income },
//   ];

//   // Metrics chart data
//   const metricsChartData = [
//     { name: "Cap Rate", value: metrics.cap_rate ? metrics.cap_rate * 100 : 0 },
//     { name: "DSCR", value: metrics.dscr || 0 },
//     { name: "CoC Return", value: metrics.coc_return ? metrics.coc_return * 100 : 0 },
//     { name: "Rent Gap %", value: metrics.rent_gap_pct || 0 },
//   ];

//   return (
//     <Box sx={{ maxWidth: 1400, mx: "auto", mt: 5, mb: 5, p: 3, bgcolor: "#fff" }}>
//       {/* Header */}
//       <Box sx={{ textAlign: "center", mb: 5 }}>
//         <Typography variant="h3" fontWeight="bold" sx={{ color: "#1976d2" }}>
//           RE Underwriting
//         </Typography>
//         <Typography variant="subtitle1" sx={{ color: "#555" }}>
//           Professional Analysis Platform
//         </Typography>
//       </Box>

//       {/* Property Overview */}
//       <Card sx={{ mb: 4, p: 3, boxShadow: 3, transition: "0.3s", "&:hover": { boxShadow: 6 } }}>
//         <CardHeader
//           title={
//             <Typography variant="h5" fontWeight="bold" sx={{ color: "#1976d2" }}>
//               {narrative_fields.property_name}
//             </Typography>
//           }
//         />
//         <CardContent>
//           <Typography sx={{ mb: 1 }}>{narrative_fields.property_address}</Typography>
//           <Typography sx={{ color: "#555" }}>
//             {narrative_fields.property_type} | {narrative_fields.total_units_or_suites} Units |{" "}
//             {narrative_fields.total_building_sqft} Sq Ft | Built {narrative_fields.year_built}
//           </Typography>
//           {narrative_fields.amenities && (
//             <Typography sx={{ mt: 1 }}>Amenities: {narrative_fields.amenities}</Typography>
//           )}
//         </CardContent>
//       </Card>

//       {/* AI Summary */}
//       {ai_summary && (
//         <Card sx={{ mb: 4, p: 3, boxShadow: 3 }}>
//           <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
//             AI Underwriting Summary
//           </Typography>
//           <Typography sx={{ color: "#555" }}>{ai_summary}</Typography>
//         </Card>
//       )}

//       {/* Financial Metrics Grid */}
//       <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
//         Financial Metrics Overview
//       </Typography>
//       <Grid container spacing={3} mb={4}>
//         <MetricCard label="Cap Rate" value={metrics.cap_rate ? (metrics.cap_rate * 100).toFixed(1) + "%" : "N/A"} />
//         <MetricCard label="DSCR" value={metrics.dscr?.toFixed(2) || "N/A"} />
//         <MetricCard label="CoC Return" value={metrics.coc_return ? (metrics.coc_return * 100).toFixed(1) + "%" : "N/A"} />
//         <MetricCard label="IRR (5yr)" value={metrics.irr_5yr ? metrics.irr_5yr.toFixed(1) + "%" : "N/A"} />
//         <MetricCard label="Rent Gap %" value={metrics.rent_gap_pct ? metrics.rent_gap_pct.toFixed(1) + "%" : "N/A"} />
//         <MetricCard label="Price per SqFt" value={metrics.price_per_sqft ? `$${metrics.price_per_sqft}` : "N/A"} />
//         <MetricCard label="Price per Unit" value={metrics.price_per_unit ? `$${metrics.price_per_unit}` : "N/A"} />
//         <MetricCard label="Break-even Occupancy" value={metrics.break_even_occupancy ? metrics.break_even_occupancy.toFixed(1) + "%" : "N/A"} />
//         <MetricCard label="NOI" value={`$${t12_summary.net_operating_income?.toLocaleString()}`} />
//       </Grid>

//       {/* T12 Line Chart */}
//       <Card sx={{ mb: 4, p: 3, boxShadow: 3 }}>
//         <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
//           T12 Financial Breakdown
//         </Typography>
//         <ResponsiveContainer width="100%" height={350}>
//           <LineChart data={t12ChartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
//             <CartesianGrid strokeDasharray="3 3" />
//             <XAxis dataKey="name" />
//             <YAxis />
//             <Tooltip />
//             <Line type="monotone" dataKey="value" stroke="#1976d2" strokeWidth={3} activeDot={{ r: 8 }} />
//           </LineChart>
//         </ResponsiveContainer>
//       </Card>

//       {/* Metrics Bar Chart */}
//       <Card sx={{ mb: 4, p: 3, boxShadow: 3 }}>
//         <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
//           Key Metrics Comparison
//         </Typography>
//         <ResponsiveContainer width="100%" height={350}>
//           <BarChart data={metricsChartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
//             <CartesianGrid strokeDasharray="3 3" />
//             <XAxis dataKey="name" />
//             <YAxis />
//             <Tooltip />
//             <Bar dataKey="value" fill="#1976d2" animationDuration={1500} />
//           </BarChart>
//         </ResponsiveContainer>
//       </Card>

//       {/* Rent Roll / Extracted Tables */}
//       {tables && tables.length > 0 && (
//         <Box sx={{ mb: 4 }}>
//           <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
//             Extracted Tables
//           </Typography>
//           {tables.map((table, idx) => (
//             <TableContainer key={idx} component={Paper} sx={{ mb: 3, boxShadow: 2 }}>
//               <Table sx={{ minWidth: 650 }} aria-label="table">
//                 <TableHead sx={{ bgcolor: "#1976d2" }}>
//                   <TableRow>
//                     {table.headers.map((h, i) => (
//                       <TableCell key={i} sx={{ color: "#fff", fontWeight: "bold" }}>
//                         {h}
//                       </TableCell>
//                     ))}
//                   </TableRow>
//                 </TableHead>
//                 <TableBody>
//                   {table.rows.map((row, rIdx) => (
//                     <TableRow key={rIdx} sx={{ "&:hover": { backgroundColor: "#e3f2fd", transition: "0.3s" } }}>
//                       {row.map((cell, cIdx) => (
//                         <TableCell key={cIdx}>{cell}</TableCell>
//                       ))}
//                     </TableRow>
//                   ))}
//                 </TableBody>
//               </Table>
//             </TableContainer>
//           ))}
//         </Box>
//       )}

//       {/* CSV Export */}
//       <Box sx={{ textAlign: "center", mt: 4 }}>
//         <CSVLink
//           data={csvData}
//           filename={`${narrative_fields.property_name}_analysis.csv`}
//           style={{
//             backgroundColor: "#1976d2",
//             color: "#fff",
//             padding: "10px 25px",
//             borderRadius: 8,
//             textDecoration: "none",
//             fontWeight: "bold",
//             transition: "0.3s",
//           }}
//         >
//           Download CSV
//         </CSVLink>
//       </Box>
//     </Box>
//   );
// }

// // Metric Card
// function MetricCard({ label, value }) {
//   return (
//     <Grid item xs={12} sm={6} md={3}>
//       <Card
//         sx={{
//           p: 3,
//           textAlign: "center",
//           boxShadow: 3,
//           transition: "0.3s",
//           "&:hover": { transform: "translateY(-5px)", boxShadow: 6 },
//         }}
//       >
//         <Typography variant="body2" color="text.secondary">
//           {label}
//         </Typography>
//         <Typography variant="h6" fontWeight="bold">
//           {value}
//         </Typography>
//       </Card>
//     </Grid>
//   );
// }










"use client";

import { useState } from "react";
import { CSVLink } from "react-csv";
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Grid,
  Paper,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
} from "@mui/material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

export default function AnalysisPage({ data }) {
  if (!data)
    return (
      <Typography
        variant="body1"
        color="text.secondary"
        align="center"
        sx={{ mt: 10, fontSize: "1.2rem" }}
      >
        No analysis yet.
      </Typography>
    );
    const { narrative_fields, metrics, t12_summary, tables, ai_summary, quick_summary } = data;

    // Use quick_summary for structured AI analysis
    const aiAnalysis = quick_summary || {};

  // CSV export data
  const csvData = [
    ["Property Name", narrative_fields.property_name],
    ["Address", narrative_fields.property_address],
    ["Type", narrative_fields.property_type],
    ["Year Built", narrative_fields.year_built],
    ["Total Units / Suites", narrative_fields.total_units_or_suites],
    ["SqFt", narrative_fields.total_building_sqft],
    ["Amenities", narrative_fields.amenities || "N/A"],
    [],
    ["Financial Metrics"],
    ["Purchase Price", "$5,000,000"],
    ["Cap Rate", metrics.cap_rate ? (metrics.cap_rate * 100).toFixed(1) + "%" : "N/A"],
    ["DSCR", metrics.dscr?.toFixed(2) || "N/A"],
    ["CoC Return", metrics.coc_return ? (metrics.coc_return * 100).toFixed(1) + "%" : "N/A"],
    ["IRR (5yr)", metrics.irr_5yr ? metrics.irr_5yr.toFixed(1) + "%" : "N/A"],
    ["Rent Gap %", metrics.rent_gap_pct ? metrics.rent_gap_pct.toFixed(1) + "%" : "N/A"],
    ["Price per SqFt", metrics.price_per_sqft ? `$${metrics.price_per_sqft}` : "N/A"],
    ["Price per Unit", metrics.price_per_unit ? `$${metrics.price_per_unit}` : "N/A"],
    ["Break-even Occupancy", metrics.break_even_occupancy ? metrics.break_even_occupancy.toFixed(1) + "%" : "N/A"],
    [],
    ["T12 Summary"],
    ["Gross Potential Rent", t12_summary.gross_potential_rent],
    ["Vacancy", t12_summary.vacancy],
    ["Effective Gross Income", t12_summary.effective_gross_income],
    ["Operating Expenses", t12_summary.operating_expenses],
    ["Net Operating Income", t12_summary.net_operating_income],
    [],
    ["AI Analysis"],
    ["Investment Recommendation", aiAnalysis.investment_recommendation || "N/A"],
    ["Key Investment Highlights", aiAnalysis.key_investment_highlights || "N/A"],
    ["Risk Considerations", aiAnalysis.risk_considerations || "N/A"],
  ];

  // Chart data for T12
  const t12ChartData = [
    { name: "Gross Rent", value: t12_summary.gross_potential_rent },
    { name: "Vacancy", value: t12_summary.vacancy },
    { name: "Operating Expenses", value: t12_summary.operating_expenses },
    { name: "Net Income", value: t12_summary.net_operating_income },
  ];

  // Metrics chart data
  const metricsChartData = [
    { name: "Cap Rate", value: metrics.cap_rate ? metrics.cap_rate * 100 : 0 },
    { name: "DSCR", value: metrics.dscr || 0 },
    { name: "CoC Return", value: metrics.coc_return ? metrics.coc_return * 100 : 0 },
    { name: "Rent Gap %", value: metrics.rent_gap_pct || 0 },
  ];

  return (
    <Box sx={{ maxWidth: 1400, mx: "auto", mt: 5, mb: 5, p: 3, bgcolor: "#fff" }}>
      {/* Header */}
      <Box sx={{ textAlign: "center", mb: 5 }}>
        <Typography variant="h3" fontWeight="bold" sx={{ color: "#1976d2" }}>
          RE Underwriting
        </Typography>
        <Typography variant="subtitle1" sx={{ color: "#555" }}>
          Professional Analysis Platform
        </Typography>
      </Box>

      {/* Property Overview */}
      <Card sx={{ mb: 4, p: 3, boxShadow: 3, transition: "0.3s", "&:hover": { boxShadow: 6 } }}>
        <CardHeader
          title={
            <Typography variant="h5" fontWeight="bold" sx={{ color: "#1976d2" }}>
              {narrative_fields.property_name}
            </Typography>
          }
        />
        <CardContent>
          <Typography sx={{ mb: 1 }}>{narrative_fields.property_address}</Typography>
          <Typography sx={{ color: "#555" }}>
            {narrative_fields.property_type} | {narrative_fields.total_units_or_suites} Units |{" "}
            {narrative_fields.total_building_sqft} Sq Ft | Built {narrative_fields.year_built}
          </Typography>
          {narrative_fields.amenities && (
            <Typography sx={{ mt: 1 }}>Amenities: {narrative_fields.amenities}</Typography>
          )}
        </CardContent>
      </Card>


      {/* AI Summary */}
{ai_summary && (
  <Card sx={{ mb: 4, p: 3, boxShadow: 3 }}>
    <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
      AI Underwriting Summary
    </Typography>
    <Typography sx={{ color: "#555" }}>{ai_summary}</Typography>
  </Card>
)}

{/* AI Analysis */}
<Card sx={{ mb: 4, p: 3, boxShadow: 3 }}>
  <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
    AI Analysis
  </Typography>
  <Typography sx={{ mb: 1 }}>
    <strong>Investment Recommendation:</strong> {aiAnalysis["Investment Recommendation"] || "N/A"}
  </Typography>
  <Typography sx={{ mb: 1 }}>
    <strong>Key Investment Highlights:</strong> {aiAnalysis["Key Investment Highlights"] || "N/A"}
  </Typography>
  <Typography sx={{ mb: 1 }}>
    <strong>Risk Considerations:</strong> {aiAnalysis["Risk Considerations"] || "N/A"}
  </Typography>
</Card>


      {/* Financial Metrics Grid */}
      <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
        Financial Metrics Overview
      </Typography>
      <Grid container spacing={3} mb={4}>
        <MetricCard label="Cap Rate" value={metrics.cap_rate ? (metrics.cap_rate * 100).toFixed(1) + "%" : "N/A"} />
        <MetricCard label="DSCR" value={metrics.dscr?.toFixed(2) || "N/A"} />
        <MetricCard label="CoC Return" value={metrics.coc_return ? (metrics.coc_return * 100).toFixed(1) + "%" : "N/A"} />
        <MetricCard label="IRR (5yr)" value={metrics.irr_5yr ? metrics.irr_5yr.toFixed(1) + "%" : "N/A"} />
        <MetricCard label="Rent Gap %" value={metrics.rent_gap_pct ? metrics.rent_gap_pct.toFixed(1) + "%" : "N/A"} />
        <MetricCard label="Price per SqFt" value={metrics.price_per_sqft ? `$${metrics.price_per_sqft}` : "N/A"} />
        <MetricCard label="Price per Unit" value={metrics.price_per_unit ? `$${metrics.price_per_unit}` : "N/A"} />
        <MetricCard label="Break-even Occupancy" value={metrics.break_even_occupancy ? metrics.break_even_occupancy.toFixed(1) + "%" : "N/A"} />
        <MetricCard label="NOI" value={`$${t12_summary.net_operating_income?.toLocaleString()}`} />
      </Grid>

      {/* T12 Line Chart */}
      <Card sx={{ mb: 4, p: 3, boxShadow: 3 }}>
        <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
          T12 Financial Breakdown
        </Typography>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={t12ChartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#1976d2" strokeWidth={3} activeDot={{ r: 8 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Metrics Bar Chart */}
      <Card sx={{ mb: 4, p: 3, boxShadow: 3 }}>
        <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
          Key Metrics Comparison
        </Typography>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={metricsChartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#1976d2" animationDuration={1500} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Rent Roll / Extracted Tables */}
      {tables && tables.length > 0 && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" fontWeight="bold" sx={{ mb: 2, color: "#1976d2" }}>
            Extracted Tables
          </Typography>
          {tables.map((table, idx) => (
            <TableContainer key={idx} component={Paper} sx={{ mb: 3, boxShadow: 2 }}>
              <Table sx={{ minWidth: 650 }} aria-label="table">
                <TableHead sx={{ bgcolor: "#1976d2" }}>
                  <TableRow>
                    {table.headers.map((h, i) => (
                      <TableCell key={i} sx={{ color: "#fff", fontWeight: "bold" }}>
                        {h}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {table.rows.map((row, rIdx) => (
                    <TableRow key={rIdx} sx={{ "&:hover": { backgroundColor: "#e3f2fd", transition: "0.3s" } }}>
                      {row.map((cell, cIdx) => (
                        <TableCell key={cIdx}>{cell}</TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ))}
        </Box>
      )}

      {/* CSV Export */}
      <Box sx={{ textAlign: "center", mt: 4 }}>
        <CSVLink
          data={csvData}
          filename={`${narrative_fields.property_name}_analysis.csv`}
          style={{
            backgroundColor: "#1976d2",
            color: "#fff",
            padding: "10px 25px",
            borderRadius: 8,
            textDecoration: "none",
            fontWeight: "bold",
            transition: "0.3s",
          }}
        >
          Download CSV
        </CSVLink>
      </Box>
    </Box>
  );
}

// Metric Card
function MetricCard({ label, value }) {
  return (
    <Grid item xs={12} sm={6} md={3}>
      <Card
        sx={{
          p: 3,
          textAlign: "center",
          boxShadow: 3,
          transition: "0.3s",
          "&:hover": { transform: "translateY(-5px)", boxShadow: 6 },
        }}
      >
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="h6" fontWeight="bold">
          {value}
        </Typography>
      </Card>
    </Grid>
  );
}

