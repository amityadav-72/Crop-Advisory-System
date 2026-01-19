const Jimp = require("jimp");

async function analyzeLeaf(imagePath) {
  const image = await Jimp.read(imagePath);

  let greenPixels = 0;
  let yellowBrownPixels = 0;
  let totalPixels = image.bitmap.width * image.bitmap.height;

  image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
    const r = this.bitmap.data[idx + 0];
    const g = this.bitmap.data[idx + 1];
    const b = this.bitmap.data[idx + 2];

    // GREEN
    if (g > 120 && g > r + 20 && g > b + 20) {
      greenPixels++;
    }

    // YELLOW / BROWN / DARK
    if (
      (r > 120 && g > 120 && b < 100) || // yellow
      (r > g && g > b) ||               // brown
      (r < 60 && g < 60 && b < 60)      // dark spots
    ) {
      yellowBrownPixels++;
    }
  });

  const greenRatio = greenPixels / totalPixels;
  const diseaseRatio = yellowBrownPixels / totalPixels;

  if (greenRatio > 0.55 && diseaseRatio < 0.15) {
    return {
      status: "Healthy 🌱",
      confidence: Math.round(greenRatio * 100),
      message: "Leaf color distribution indicates healthy crop"
    };
  }

  return {
    status: "Disease Detected ⚠️",
    confidence: Math.round(diseaseRatio * 100),
    message: "Abnormal leaf color patterns detected"
  };
}

module.exports = analyzeLeaf;
