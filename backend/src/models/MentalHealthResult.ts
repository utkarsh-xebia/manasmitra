import mongoose, { Schema, Document, Types } from 'mongoose';

export type SeverityLabel = 'Minimal' | 'Mild' | 'Moderate' | 'Moderately Severe' | 'Severe';

export interface IMentalHealthResult extends Document {
  userId: Types.ObjectId;
  severityLabel: SeverityLabel;
  confidenceScore: number;
  totalScore: number;
  stressLevel: number;
  moodScore: number;
  workLifeBalance: number;
  recommendations: string[];
  modelUsed: string;
  createdAt: Date;
  updatedAt: Date;
}

const MentalHealthResultSchema = new Schema<IMentalHealthResult>(
  {
    userId: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: true,
      index: true,
    },
    severityLabel: {
      type: String,
      enum: ['Minimal', 'Mild', 'Moderate', 'Moderately Severe', 'Severe'],
      required: true,
    },
    confidenceScore: {
      type: Number,
      required: true,
      min: 0,
      max: 1,
    },
    totalScore: {
      type: Number,
      required: true,
    },
    stressLevel: {
      type: Number,
      required: true,
    },
    moodScore: {
      type: Number,
      required: true,
    },
    workLifeBalance: {
      type: Number,
      required: true,
    },
    recommendations: {
      type: [String],
      required: true,
    },
    modelUsed: {
      type: String,
      required: true,
      default: 'phq9_best_model',
    },
  },
  {
    timestamps: true,
  }
);

export const MentalHealthResult = mongoose.model<IMentalHealthResult>(
  'MentalHealthResult',
  MentalHealthResultSchema
);
