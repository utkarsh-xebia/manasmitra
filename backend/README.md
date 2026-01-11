# Manasmitra Backend API

Express + TypeScript backend for the Employee Wellbeing & Burnout Assessment platform.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

3. Update `.env` with your MongoDB connection string:
```
MONGODB_URI=mongodb://localhost:27017/manasmitra
JWT_SECRET=your-secret-key-change-in-production
PORT=5000
```

4. Make sure MongoDB is running locally or update `MONGODB_URI` to your MongoDB instance.

5. Run in development mode:
```bash
npm run dev
```

Or build and run in production:
```bash
npm run build
npm start
```

## API Endpoints

### Questionnaire

- `POST /api/questionnaire/submit` - Submit PHQ-9 questionnaire (Employee only, requires authentication)
- `GET /api/questionnaire/results` - Get user's questionnaire results (Employee only)
- `GET /api/questionnaire/latest` - Get user's latest mental health result (Employee only)

### Health Check

- `GET /health` - Server health check

## Database Models

- **User**: User authentication and profile
- **QuestionnaireResponse**: PHQ-9 questionnaire responses
- **MentalHealthResult**: AI prediction results

## Architecture

- **Models**: Mongoose schemas for database entities
- **Routes**: Express route handlers
- **Middleware**: Authentication and authorization middleware
- **Services**: Business logic (AI inference, etc.)
- **Config**: Database and server configuration
