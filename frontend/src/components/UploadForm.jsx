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
import { CloudUpload, InsertDriveFile, CheckCircle } from "@mui/icons-material";

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
        width: "220%",
        height: "90vh",
        bgcolor: "#e3f2fd",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        p: 3,
      }}
    >
      <Card
        elevation={12}
        sx={{
          width: "100%",
          maxWidth: 900, // max width for large screens
          borderRadius: 4,
          boxShadow: "0px 15px 30px rgba(0,0,0,0.15)",
          transition: "0.3s",
          "&:hover": { boxShadow: "0px 20px 40px rgba(0,0,0,0.25)" },
        }}
      >
        <CardHeader
          title={
            <Typography
              variant="h3"
              align="center"
              sx={{
                fontWeight: "bold",
                background: "linear-gradient(90deg, #1976d2, #42a5f5)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              Upload PDF for Analysis
            </Typography>
          }
        />
        <CardContent sx={{ px: { xs: 2, sm: 4, md: 6 }, py: 4 }}>
          {/* Dropzone */}
          <Paper
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            variant="outlined"
            sx={{
              p: { xs: 4, sm: 6 },
              textAlign: "center",
              borderStyle: "dashed",
              borderColor: "#42a5f5",
              cursor: "pointer",
              mb: 4,
              position: "relative",
              borderRadius: 3,
              transition: "0.3s",
              "&:hover": {
                borderColor: "#1976d2",
                backgroundColor: "#bbdefb",
              },
            }}
          >
            <CloudUpload sx={{ fontSize: 80, color: "#1976d2", mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
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
            <List
              sx={{
                mb: 4,
                maxHeight: 250,
                overflowY: "auto",
                bgcolor: "#f9f9f9",
                borderRadius: 2,
                px: 1,
                py: 0.5,
              }}
            >
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
                    <InsertDriveFile sx={{ color: "#1976d2" }} />
                  </ListItemIcon>
                  <ListItemText
                    primary={file.name}
                    secondary={`${(file.size / 1024).toFixed(2)} KB`}
                  />
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
                sx={{
                  height: 14,
                  borderRadius: 6,
                  backgroundColor: "#bbdefb",
                  "& .MuiLinearProgress-bar": {
                    background: "linear-gradient(90deg, #1976d2, #42a5f5)",
                  },
                }}
              />
              <Typography
                variant="body2"
                color="text.secondary"
                align="center"
                sx={{ mt: 1, fontWeight: "bold" }}
              >
                {progress}%
              </Typography>
            </Box>
          )}

          {/* Submit Button */}
          <Button
            variant="contained"
            fullWidth
            onClick={handleSubmit}
            disabled={loading}
            sx={{
              py: 2,
              fontWeight: "bold",
              fontSize: "1.2rem",
              background: "linear-gradient(90deg, #1976d2, #42a5f5)",
              "&:hover": {
                background: "linear-gradient(90deg, #1565c0, #1e88e5)",
                transform: "scale(1.03)",
              },
            }}
          >
            {loading ? "Analyzing..." : "Upload & Analyze"}
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
}
