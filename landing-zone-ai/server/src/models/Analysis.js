const mongoose = require('mongoose');

const AnalysisSchema = new mongoose.Schema({
  imageUrl: {
    type: String,
    required: true
  },
  heatmapUrl: {
    type: String,
  },
  score: {
    type: Number,
  },
  stats: {
    type: Map,
    of: Number
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Analysis', AnalysisSchema);
