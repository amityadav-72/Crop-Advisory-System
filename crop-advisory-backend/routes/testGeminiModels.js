require("dotenv").config();
const { GoogleGenerativeAI } = require("@google/generative-ai");

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

(async () => {
  try {
    const models = await genAI.listModels();
    console.log(
      models.models.map(m => ({
        name: m.name,
        methods: m.supportedGenerationMethods
      }))
    );
  } catch (err) {
    console.error("Error listing models:", err.message);
  }
})();
