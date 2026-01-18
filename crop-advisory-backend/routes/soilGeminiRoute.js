const express = require("express");
const multer = require("multer");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const router = express.Router();
const upload = multer({ dest: "uploads/" });

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

/* POST: Analyze Soil (TEXT-ONLY GEMINI) */
router.post("/analyze-soil", upload.single("soilImage"), async (req, res) => {
  try {
    // We accept image but DO NOT send it to Gemini
    console.log("📥 Soil image received (not sent to Gemini)");

    const model = genAI.getGenerativeModel({ model: "gemini-pro" });

    const prompt = `
A soil image has been captured from a farm field.

Based on general soil analysis principles, generate a soil health report including:
- Soil type
- Moisture level
- Nutrient condition
- Suitable crops
- Fertilizer recommendation

Keep it concise and practical.
`;

    const result = await model.generateContent(prompt);

    res.json({
      report: result.response.text()
    });

  } catch (err) {
    console.error("❌ Gemini Error:", err.message);
    res.status(500).json({ error: "Failed to generate soil report" });
  }
});

module.exports = router;
