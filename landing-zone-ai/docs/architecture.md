# Architecture

## Overview
The system consists of three main components:
1. **Client**: React-based frontend for user interaction.
2. **Server**: Node.js/Express backend for API management and orchestration.
3. **Python AI**: Flask-based microservice for Deep Learning inference.

## Data Flow
1. User uploads image via Client.
2. Client sends image to Server.
3. Server saves image and forwards path to Python AI.
4. Python AI processes image (inference + heatmap).
5. Python AI returns results to Server.
6. Server saves results to MongoDB and responds to Client.
7. Client displays original image, heatmap, and stats.
