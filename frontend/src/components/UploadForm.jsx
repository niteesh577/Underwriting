"use client";

import { useState } from "react";
import { uploadFiles } from "../api";
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
} from "@mui/material";
import { CloudUpload, Description, CheckCircle } from "@mui/icons-material";

export default function UploadForm({ onAnalysisComplete }) {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleFileChange = (e) => setFiles(Array.from(e.target.files));
  const handleDrop = (e) => {
    e.preventDefault();
    setFiles((prev) => [...prev, ...Array.from(e.dataTransfer.files)]);
  };
  const handleDragOver = (e) => e.preventDefault();

  const handleSubmit = async () => {
    if (!files.length) return;
    setLoading(true);
    setProgress(0);

    try {
      const interval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 100));
      }, 100);

      const result = await uploadFiles(files, { purchase_price: 5000000 });
      clearInterval(interval);
      setProgress(100);
      onAnalysisComplete(result);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        width: "100%", // full page width
        minHeight: "100vh", // full page height
        bgcolor: "#f0f2f5",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        px: 2,
      }}
    >
      {/* Upload Form Card (medium size, centered) */}
      <Card
        elevation={10}
        sx={{
          width: { xs: "90%", sm: "70%", md: 500 }, // medium fixed width
          maxWidth: "100%",
          bgcolor: "#fff",
          borderRadius: 3,
          boxShadow: 8,
          transition: "0.3s",
          "&:hover": { boxShadow: 12 },
          py: 5,
          px: 4,
        }}
      >
        <CardHeader
          title={
            <Typography
              variant="h5"
              align="center"
              sx={{ color: "#1976d2", fontWeight: "bold" }}
            >
              Upload PDF for Analysis
            </Typography>
          }
        />
        <CardContent>
          {/* Dropzone */}
          <Paper
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            variant="outlined"
            sx={{
              p: 6,
              textAlign: "center",
              borderStyle: "dashed",
              borderColor: "#1976d2",
              cursor: "pointer",
              mb: 4,
              position: "relative",
              borderRadius: 3,
              transition: "0.3s",
              "&:hover": { borderColor: "#0d47a1", backgroundColor: "#e3f2fd" },
            }}
          >
            <CloudUpload sx={{ fontSize: 70, color: "#1976d2", mb: 2 }} />
            <Typography variant="body1" color="text.secondary">
              Drag & Drop your PDF files here <br /> or click to select
            </Typography>
            <input
              type="file"
              multiple
              onChange={handleFileChange}
              style={{
                position: "absolute",
                width: "100%",
                height: "100%",
                top: 0,
                left: 0,
                opacity: 0,
                cursor: "pointer",
              }}
            />
          </Paper>

          {/* File Preview */}
          {files.length > 0 && (
            <List sx={{ mb: 4 }}>
              {files.map((file, idx) => (
                <ListItem
                  key={idx}
                  sx={{
                    mb: 1,
                    p: 2,
                    border: "1px solid #e0e0e0",
                    borderRadius: 2,
                    transition: "0.3s",
                    "&:hover": { backgroundColor: "#e3f2fd" },
                  }}
                >
                  <ListItemIcon>
                    <Description sx={{ color: "#1976d2" }} />
                  </ListItemIcon>
                  <ListItemText primary={file.name} />
                  <CheckCircle sx={{ color: "success.main" }} />
                </ListItem>
              ))}
            </List>
          )}

          {/* Progress */}
          {loading && (
            <Box sx={{ width: "100%", mb: 3 }}>
              <LinearProgress
                variant="determinate"
                value={progress}
                sx={{ height: 12, borderRadius: 6 }}
              />
              <Typography
                variant="body2"
                color="text.secondary"
                align="center"
                sx={{ mt: 1 }}
              >
                {progress}%
              </Typography>
            </Box>
          )}

          {/* Submit Button */}
          <Button
            variant="contained"
            color="primary"
            fullWidth
            onClick={handleSubmit}
            disabled={loading}
            sx={{
              py: 2,
              fontWeight: "bold",
              transition: "0.3s",
              "&:hover": { backgroundColor: "#0d47a1", transform: "scale(1.03)" },
            }}
          >
            {loading ? "Analyzing..." : "Upload & Analyze"}
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
}
