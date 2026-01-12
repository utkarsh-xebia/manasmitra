import mongoose, { Schema, Document, Types } from 'mongoose';

export interface IUser extends Document {
  name: string;
  email: string;
  password: string;
  role: 'hr' | 'manager' | 'employee';
  age?: number;
  gender?: string;
  department?: string;
  reportingManager?: Types.ObjectId;
  status: 'active' | 'inactive';
  isDeleted: boolean;
  createdAt: Date;
  updatedAt: Date;
}

const UserSchema = new Schema<IUser>(
  {
    name: {
      type: String,
      required: true,
      trim: true,
    },
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
    },
    password: {
      type: String,
      required: false, // Allow HR to create stubs without passwords
    },
    role: {
      type: String,
      enum: ['hr', 'manager', 'employee'],
      required: true,
    },
    age: {
      type: Number,
    },
    gender: {
      type: String,
    },
    department: {
      type: String,
      trim: true,
    },
    reportingManager: {
      type: Schema.Types.ObjectId,
      ref: 'User',
    },
    status: {
      type: String,
      enum: ['active', 'inactive'],
      default: 'active',
    },
    isDeleted: {
      type: Boolean,
      default: false,
    },
    needsPasswordReset: {
      type: Boolean,
      default: false,
    },
  },
  {
    timestamps: true,
  }
);

export const User = mongoose.model<IUser>('User', UserSchema);
