const express = require('express');
const cors = require('cors');
const path = require('path');
const analysisRoutes = require('./routes/analysisRoutes');
const { errorHandler } = require('./middlewares/errorHandler');

const app = express();

app.use(cors());
app.use(express.json());
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));
app.use('/heatmaps', express.static(path.join(__dirname, '../heatmaps')));

app.use('/api/analysis', analysisRoutes);

app.use(errorHandler);

module.exports = app;
