const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const { spawn } = require("child_process");
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

/* ---------- MODELS ---------- */
const Disease = require("./models/Disease");

/* ---------- ROUTES ---------- */
app.get("/", (req, res) => {
  res.send("🌱 Crop Advisory Backend Running");
});

app.use("/api/auth", require("./routes/authRoutes"));
app.use("/api/soil", require("./routes/soilGeminiRoute"));

/* ---------- AI DISEASE DETECTION ---------- */
app.post("/detect-disease", upload.single("cropImage"), (req, res) => {
  console.log("📥 /detect-disease HIT");

  if (!req.file) {
    return res.status(400).json({ message: "No image uploaded" });
  }

  console.log("✅ Image saved at:", req.file.path);

  const python = spawn("python", ["ai/predict.py", req.file.path]);
  let output = "";

  python.stdout.on("data", (data) => {
    output += data.toString();
  });

  python.stderr.on("data", (data) => {
    console.error("🐍 Python Error:", data.toString());
  });

  python.on("close", async () => {
    if (!output) {
      return res.status(500).json({ message: "AI model returned no output" });
    }

    const predictedRaw = output.trim();
    console.log("🧠 Raw Prediction:", predictedRaw);

    // ---------------- NORMALIZE AI OUTPUT ----------------
    const [rawCrop, rawDisease] = predictedRaw.split("___");

    const cropType = rawCrop
      .replace(/_/g, " ")
      .replace(/\(.*?\)/g, "")
      .trim();

    const diseaseName = rawDisease
      .replace(/_/g, " ")
      .trim();

    console.log("🌾 Crop:", cropType);
    console.log("🦠 Disease:", diseaseName);

    /* 🌱 HEALTHY CASE */
    if (rawDisease.toLowerCase().includes("healthy")) {
      return res.json({
        disease: `${cropType} - Healthy 🌱`,
        message: "Your crop is healthy. No treatment required.",
      });
    }

    try {
      // 🔎 FLEXIBLE DATABASE SEARCH
      const disease = await Disease.findOne({
        diseaseName: { $regex: diseaseName, $options: "i" },
        cropType: { $regex: cropType, $options: "i" },
      });

      if (disease) {
        return res.json({
          disease: `${cropType} - ${diseaseName}`,
          medicine: disease.medicineName,
          solution: disease.instructions,
          cropType: disease.cropType,
        });
      }

      // fallback if crop-specific entry not found
      const fallback = await Disease.findOne({
        diseaseName: { $regex: diseaseName, $options: "i" },
      });

      if (fallback) {
        return res.json({
          disease: `${cropType} - ${diseaseName}`,
          medicine: fallback.medicineName,
          solution: fallback.instructions,
          cropType: fallback.cropType,
        });
      }

      return res.json({
        disease: `${cropType} - ${diseaseName}`,
        message: "Disease not found in database",
      });
    } catch (err) {
      console.error("❌ Database Error:", err);
      return res.status(500).json({ message: "Database error" });
    }
  });
});

/* ---------- SERVER ---------- */
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
