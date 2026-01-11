/**
 * AI Inference Service
 * Handles loading and running predictions using pre-trained PHQ-9 model
 */

import { SeverityLabel } from '../models/MentalHealthResult.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface PredictionInput {
  answers: number[]; // Array of 9 PHQ-9 answers (0-3)
  age?: number;
  gender?: string;
  sleepQuality?: string;
  studyPressure?: string;
  financialPressure?: string;
}

interface PredictionResult {
  severityLabel: SeverityLabel;
  confidenceScore: number;
  totalScore: number;
  stressLevel: number;
  moodScore: number;
  workLifeBalance: number;
  recommendations: string[];
  probabilities?: Record<string, number>;
}

/**
 * Run PHQ-9 prediction using Python model
 */
export async function predictPHQ9Severity(input: PredictionInput): Promise<PredictionResult> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [path.join(__dirname, 'predict.py')]);

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python script error:', errorOutput);
        reject(new Error(`Inference failed with code ${code}: ${errorOutput}`));
        return;
      }

      try {
        const result = JSON.parse(output);
        if (result.error) {
          reject(new Error(result.error));
        } else {
          resolve(result);
        }
      } catch (err) {
        console.error('Failed to parse Python output:', output);
        reject(new Error('Failed to parse inference result'));
      }
    });

    // Write input to stdin
    pythonProcess.stdin.write(JSON.stringify(input));
    pythonProcess.stdin.end();
  });
}

