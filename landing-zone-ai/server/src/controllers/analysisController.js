const Analysis = require('../models/Analysis');
const pythonService = require('../services/pythonService');
const logger = require('../utils/logger');

exports.analyzeImage = async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No image uploaded' });
    }

    const imagePath = req.file.path;
    logger.info(`Processing image: ${imagePath}`);

    // Call Python Service
    const aiResult = await pythonService.processImage(imagePath);

    const newAnalysis = await Analysis.create({
      imageUrl: `/uploads/${req.file.filename}`,
      heatmapUrl: aiResult.heatmapUrl,
      score: aiResult.score,
      stats: aiResult.stats
    });

    res.status(201).json({
      success: true,
      data: newAnalysis
    });

  } catch (error) {
    logger.error(error);
    next(error);
  }
};

exports.getHistory = async (req, res, next) => {
  try {
    const history = await Analysis.find().sort({ createdAt: -1 });
    res.status(200).json({
      success: true,
      data: history
    });
  } catch (error) {
    next(error);
  }
};
