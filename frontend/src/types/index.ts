export type Role = 'hr' | 'manager' | 'employee';

export interface KPIMetric {
  label: string;
  value: string | number;
  change?: number;
  trend?: 'up' | 'down' | 'neutral';
  icon?: string;
  color?: string;
}

export interface ChartDataPoint {
  name: string;
  value: number;
  [key: string]: string | number;
}

export interface DepartmentHealth {
  name: string;
  score: number;
  employees: number;
  atRisk: number;
}

export interface TeamMember {
  id: string;
  name: string;
  wellbeingScore: number;
  burnoutRisk: 'low' | 'medium' | 'high';
  engagementScore: number;
}

export interface WellbeingTrend {
  month: string;
  score: number;
  workload?: number;
}

export interface Alert {
  id: string;
  type: 'warning' | 'error' | 'info';
  title: string;
  message: string;
  department?: string;
}

export interface WellbeingProgram {
  id: string;
  name: string;
  description: string;
  enrolled: boolean;
  startDate?: string;
}

export interface MentalHealthResult {
  _id: string;
  userId: string;
  severityLabel: 'Minimal' | 'Mild' | 'Moderate' | 'Moderately Severe' | 'Severe';
  confidenceScore: number;
  totalScore: number;
  stressLevel: number;
  moodScore: number;
  workLifeBalance: number;
  recommendations: string[];
  modelUsed: string;
  createdAt: string;
  updatedAt: string;
}