import express, { Router } from 'express';
import { authenticate, requireRole, AuthRequest } from '../middleware/auth.js';
import { QuestionnaireResponse } from '../models/QuestionnaireResponse.js';
import { MentalHealthResult } from '../models/MentalHealthResult.js';
import { predictPHQ9Severity } from '../services/aiInference.js';

const router = Router();

/**
 * POST /api/questionnaire/submit
 * Submit PHQ-9 questionnaire (Employee only)
 */
router.post(
  '/submit',
  authenticate,
  requireRole(['employee']),
  async (req: AuthRequest, res) => {
    try {
      const { answers, age, gender, sleepQuality, studyPressure, financialPressure } = req.body;
      const userId = req.userId!;

      // Validate input
      if (!Array.isArray(answers) || answers.length !== 9) {
        res.status(400).json({ error: 'Answers must be an array of 9 numbers (0-3)' });
        return;
      }

      // Calculate total score
      const totalScore = answers.reduce((sum: number, val: number) => sum + val, 0);

      // Save questionnaire response
      const questionnaireResponse = new QuestionnaireResponse({
        userId,
        answers,
        totalScore,
        age,
        gender,
        sleepQuality,
        studyPressure,
        financialPressure,
      });
      await questionnaireResponse.save();

      // Run AI inference
      const predictionResult = await predictPHQ9Severity({ 
        answers, 
        age, 
        gender, 
        sleepQuality, 
        studyPressure, 
        financialPressure 
      });

      // Save mental health result
      const mentalHealthResult = new MentalHealthResult({
        userId,
        severityLabel: predictionResult.severityLabel,
        confidenceScore: predictionResult.confidenceScore,
        totalScore: predictionResult.totalScore,
        stressLevel: predictionResult.stressLevel,
        moodScore: predictionResult.moodScore,
        workLifeBalance: predictionResult.workLifeBalance,
        recommendations: predictionResult.recommendations,
        modelUsed: 'phq9_best_model',
      });
      await mentalHealthResult.save();

      // Return result
      res.status(200).json({
        success: true,
        result: mentalHealthResult,
      });
    } catch (error) {
      console.error('Questionnaire submission error:', error);
      res.status(500).json({ error: 'Failed to process questionnaire' });
    }
  }
);

/**
 * GET /api/questionnaire/results
 * Get user's questionnaire results (Employee only)
 */
router.get('/results', authenticate, requireRole(['employee']), async (req: AuthRequest, res) => {
  try {
    const userId = req.userId!;

    const results = await MentalHealthResult.find({ userId })
      .sort({ createdAt: -1 })
      .limit(10)
      .lean();

    res.status(200).json({ success: true, results });
  } catch (error) {
    console.error('Error fetching results:', error);
    res.status(500).json({ error: 'Failed to fetch results' });
  }
});

/**
 * GET /api/questionnaire/latest
 * Get user's latest mental health result (Employee only)
 */
router.get('/latest', authenticate, requireRole(['employee']), async (req: AuthRequest, res) => {
  try {
    const userId = req.userId!;

    const latestResult = await MentalHealthResult.findOne({ userId })
      .sort({ createdAt: -1 })
      .lean();

    if (!latestResult) {
      res.status(404).json({ error: 'No results found' });
      return;
    }

    res.status(200).json({ success: true, result: latestResult });
  } catch (error) {
    console.error('Error fetching latest result:', error);
    res.status(500).json({ error: 'Failed to fetch latest result' });
  }
});

export default router;
