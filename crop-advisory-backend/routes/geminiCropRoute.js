const express = require("express");
const multer = require("multer");
const fs = require("fs");

const router = express.Router();
const upload = multer({ dest: "uploads/" });

router.post("/crop-advisory", upload.single("image"), async (req, res) => {
  try {
    // ✅ DEFENSIVE CHECK
    if (!req.file) {
      return res.status(400).json({
        message: "Image not received. Please upload an image file."
      });
    }

    const { crop } = req.body;
    const imagePath = req.file.path;

    const imageBase64 = fs.readFileSync(imagePath, "base64");
    fs.unlinkSync(imagePath);

    const prompt = `
You are an agriculture expert.

Crop: ${crop}

Analyze the crop leaf image and respond ONLY in JSON:
{
  "issue": "",
  "cause": "",
  "solution": "",
  "fertilizer": "",
  "weather": ""
}
`;

    const response = await fetch(
      "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=" +
        process.env.GEMINI_API_KEY,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [
            {
              role: "user",
              parts: [
                { text: prompt },
                {
                  inline_data: {
                    mime_type: "image/jpeg",
                    data: imageBase64
                  }
                }
              ]
            }
          ]
        })
      }
    );

    const data = await response.json();

    const text =
      data?.candidates?.[0]?.content?.parts?.[0]?.text;

    if (!text) {
      return res.status(500).json({ message: "Gemini response invalid" });
    }

    const cleanJSON = JSON.parse(
      text.replace(/```json|```/g, "").trim()
    );

    res.json({ crop, ...cleanJSON });

  } catch (err) {
    console.error("GEMINI ERROR:", err);
    res.status(500).json({ message: "Gemini AI failed" });
  }
});

module.exports = router;
