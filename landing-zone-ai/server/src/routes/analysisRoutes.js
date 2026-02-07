const express = require('express');
const router = express.Router();
const upload = require('../middlewares/uploadMiddleware');
const { analyzeImage, getHistory } = require('../controllers/analysisController');

router.post('/upload', upload.single('image'), analyzeImage);
router.get('/history', getHistory);

module.exports = router;
