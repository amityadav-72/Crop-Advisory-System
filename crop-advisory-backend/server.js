const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const axios = require("axios");
require("dotenv").config();

const app = express();

/* ---------- MIDDLEWARE ---------- */
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use("/uploads", express.static("uploads"));

/* ---------- MONGODB ---------- */
mongoose
  .connect(process.env.MONGO_URI || "mongodb://127.0.0.1:27017/cropDB")
  .then(() => console.log("✅ MongoDB Connected"))
  .catch((err) => {
    console.error("❌ MongoDB Error:", err.message);
    process.exit(1);
  });

/* ---------- MULTER ---------- */
const storage = multer.diskStorage({
  destination: "./uploads/",
  filename: (req, file, cb) => {
    cb(null, "crop-" + Date.now() + path.extname(file.originalname));
  },
});
const upload = multer({ storage });

/* ---------- ROOT ---------- */
app.get("/", (req, res) => {
  res.send("🌱 Crop Advisory Backend Running (Plant.id API v2)");
});

/* ---------- 🌿 PLANT.ID HEALTH API (NO LOCAL ML) ---------- */
app.post("/detect-disease", upload.single("cropImage"), async (req, res) => {
  console.log("📥 /detect-disease HIT");

  if (!req.file) {
    return res.status(400).json({ message: "No image uploaded" });
  }

  try {
    // Convert image to base64
    const imageBase64 = fs.readFileSync(req.file.path, {
      encoding: "base64",
    });

    // Call Plant.id API v2
    const response = await axios.post(
      "https://api.plant.id/v2/health_assessment",
      {
        images: [imageBase64],
        health: "only",
      },
      {
        headers: {
          "Content-Type": "application/json",
          "Api-Key": process.env.PLANT_HEALTH_API_KEY,
        },
        timeout: 20000,
      }
    );

    // Delete uploaded image
    fs.unlinkSync(req.file.path);

    const assessment = response.data.health_assessment;

    // 🌱 HEALTHY CASE
    if (assessment.is_healthy) {
      return res.json({
        disease: "Healthy 🌱",
        confidence: (assessment.is_healthy_probability * 100).toFixed(2),
        message: "Your crop is healthy. No disease detected.",
      });
    }

    // 🦠 BIOTIC DISEASE
    let diseaseName = "No specific disease detected";
    let diseaseConfidence = null;

    if (assessment.diseases && assessment.diseases.length > 0) {
      diseaseName = assessment.diseases[0].name;
      diseaseConfidence = (
        assessment.diseases[0].probability * 100
      ).toFixed(2);
    }

    // 💧 ABIOTIC STRESS (water, nutrients, etc.)
    let abioticIssue = null;
    let abioticConfidence = null;

    if (
      assessment.abiotic_stresses &&
      assessment.abiotic_stresses.length > 0
    ) {
      abioticIssue = assessment.abiotic_stresses[0].name;
      abioticConfidence = (
        assessment.abiotic_stresses[0].probability * 100
      ).toFixed(2);
    }

    // ✅ FINAL RESPONSE
    return res.json({
      disease: diseaseName,
      diseaseConfidence: diseaseConfidence,
      abioticIssue: abioticIssue,
      abioticConfidence: abioticConfidence,
      message: "Analysis completed",
    });

  } catch (error) {
    console.error("❌ Plant.id API Error:");
    console.error(error.response?.data || error.message);

    return res.status(500).json({
      message: "Plant.id Health API request failed",
    });
  }
});

/* ---------- SERVER ---------- */
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
