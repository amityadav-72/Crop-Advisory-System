const mongoose = require('mongoose');

// Define the schema once with all necessary fields
const diseaseSchema = new mongoose.Schema({
    diseaseName: { 
        type: String, 
        required: true, 
        unique: true  // Ensures you don't have duplicate disease entries
    },
    symptoms: String,
    medicineName: [String], // Array allows for multiple medicine recommendations
    instructions: String,   // Step-by-step solution for the farmer
    cropType: String        // Helps filter results by crop (Rice, Wheat, etc.)
});

// Export the model so server.js can use it
module.exports = mongoose.model('Disease', diseaseSchema);