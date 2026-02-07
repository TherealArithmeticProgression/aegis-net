# API Specification

## Server API

### POST /api/analysis/upload
- **Body**: `multipart/form-data` with `image` file.
- **Response**: JSON object with analysis results.

### GET /api/analysis/history
- **Response**: Array of past analysis records.

## Python AI API

### POST /predict
- **Body**: `multipart/form-data` with `image` file.
- **Response**: JSON object with `score`, `heatmapUrl`, and `stats`.
