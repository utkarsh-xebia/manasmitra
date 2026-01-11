import mongoose, { Schema, Document, Types } from 'mongoose';

export interface IQuestionnaireResponse extends Document {
  userId: Types.ObjectId;
  answers: number[]; // Array of 9 numbers (0-3)
  difficultyLevel?: string;
  totalScore: number;
  age?: number;
  gender?: string;
  sleepQuality?: string;
  studyPressure?: string;
  financialPressure?: string;
  createdAt: Date;
  updatedAt: Date;
}

const QuestionnaireResponseSchema = new Schema<IQuestionnaireResponse>(
  {
    userId: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: true,
      index: true,
    },
    answers: {
      type: [Number],
      required: true,
      validate: {
        validator: (arr: number[]) => arr.length === 9 && arr.every((val) => val >= 0 && val <= 3),
        message: 'Answers must be an array of 9 numbers between 0 and 3',
      },
    },
    difficultyLevel: {
      type: String,
    },
    totalScore: {
      type: Number,
      required: true,
      min: 0,
      max: 27,
    },
    age: Number,
    gender: String,
    sleepQuality: String,
    studyPressure: String,
    financialPressure: String,
  },
  {
    timestamps: true,
  }
);

export const QuestionnaireResponse = mongoose.model<IQuestionnaireResponse>(
  'QuestionnaireResponse',
  QuestionnaireResponseSchema
);
