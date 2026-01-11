import { DepartmentHealth, TeamMember, WellbeingTrend, Alert, WellbeingProgram, ChartDataPoint } from '../types';

// HR Mock Data
export const hrMockData = {
  companyWellbeingScore: 72,
  employeesAtRisk: 23,
  activePrograms: 8,
  surveyResponseRate: 78,
  burnoutDistribution: [
    { name: 'Low', value: 145, color: '#10b981' },
    { name: 'Medium', value: 78, color: '#f59e0b' },
    { name: 'High', value: 32, color: '#ef4444' },
  ] as ChartDataPoint[],
  departmentHealth: [
    { name: 'Engineering', score: 68, employees: 45, atRisk: 8 },
    { name: 'Sales', score: 75, employees: 32, atRisk: 5 },
    { name: 'Marketing', score: 80, employees: 28, atRisk: 3 },
    { name: 'Support', score: 65, employees: 24, atRisk: 6 },
    { name: 'Operations', score: 70, employees: 38, atRisk: 7 },
    { name: 'HR', score: 85, employees: 12, atRisk: 1 },
  ] as DepartmentHealth[],
  monthlyTrend: [
    { month: 'Jan', score: 70 },
    { month: 'Feb', score: 71 },
    { month: 'Mar', score: 69 },
    { month: 'Apr', score: 72 },
    { month: 'May', score: 73 },
    { month: 'Jun', score: 72 },
  ] as WellbeingTrend[],
  alerts: [
    {
      id: '1',
      type: 'warning',
      title: 'High Risk Department',
      message: 'Engineering department shows 18% at-risk employees',
      department: 'Engineering',
    },
    {
      id: '2',
      type: 'warning',
      title: 'Low Response Rate',
      message: 'Support team has 45% survey response rate',
      department: 'Support',
    },
  ] as Alert[],
};

// Manager Mock Data
export const managerMockData = {
  teamWellbeingScore: 68,
  teamMembersAtRisk: 3,
  engagementScore: 75,
  recentSurveyResult: 72,
  teamMembers: [
    { id: '1', name: 'Sarah Chen', wellbeingScore: 75, burnoutRisk: 'low' as const, engagementScore: 82 },
    { id: '2', name: 'Michael Park', wellbeingScore: 65, burnoutRisk: 'medium' as const, engagementScore: 70 },
    { id: '3', name: 'Emily Johnson', wellbeingScore: 70, burnoutRisk: 'low' as const, engagementScore: 78 },
    { id: '4', name: 'David Kim', wellbeingScore: 58, burnoutRisk: 'high' as const, engagementScore: 65 },
    { id: '5', name: 'Lisa Wang', wellbeingScore: 72, burnoutRisk: 'low' as const, engagementScore: 80 },
    { id: '6', name: 'James Brown', wellbeingScore: 62, burnoutRisk: 'medium' as const, engagementScore: 68 },
  ] as TeamMember[],
  workloadVsWellbeing: [
    { month: 'Jan', workload: 75, score: 70 },
    { month: 'Feb', workload: 78, score: 68 },
    { month: 'Mar', workload: 82, score: 65 },
    { month: 'Apr', workload: 80, score: 67 },
    { month: 'May', workload: 75, score: 69 },
    { month: 'Jun', workload: 73, score: 68 },
  ] as WellbeingTrend[],
  insights: [
    '2 employees showing early burnout signs',
    'Consider reducing workload for high-risk team members',
    'Team engagement is above average',
  ],
};

// Employee Mock Data
export const employeeMockData = {
  personalWellbeingScore: 72,
  stressLevel: 4.2,
  moodScore: 7.5,
  workLifeBalance: 6.8,
  stressTrend: [
    { month: 'Jan', score: 4.5 },
    { month: 'Feb', score: 4.2 },
    { month: 'Mar', score: 4.8 },
    { month: 'Apr', score: 4.0 },
    { month: 'May', score: 4.3 },
    { month: 'Jun', score: 4.2 },
  ] as WellbeingTrend[],
  moodTrend: [
    { month: 'Jan', score: 7.0 },
    { month: 'Feb', score: 7.2 },
    { month: 'Mar', score: 7.1 },
    { month: 'Apr', score: 7.5 },
    { month: 'May', score: 7.6 },
    { month: 'Jun', score: 7.5 },
  ] as WellbeingTrend[],
  recommendations: [
    { id: '1', action: 'Get 7-8 hours of sleep', status: 'active' },
    { id: '2', action: 'Take regular breaks every 2 hours', status: 'active' },
    { id: '3', action: '30 minutes of exercise daily', status: 'pending' },
    { id: '4', action: 'Practice mindfulness meditation', status: 'pending' },
  ],
  activePrograms: [
    { id: '1', name: 'Mental Health Awareness', description: 'Weekly workshops on mental health', enrolled: true, startDate: '2024-01-15' },
    { id: '2', name: 'Flexible Working Hours', description: 'Flexible schedule program', enrolled: true, startDate: '2024-02-01' },
    { id: '3', name: 'Fitness Challenge', description: 'Monthly fitness goals and tracking', enrolled: false },
  ] as WellbeingProgram[],
  upcomingSurveys: [
    { id: '1', name: 'Monthly Wellbeing Check-in', date: '2024-07-15' },
    { id: '2', name: 'Quarterly Engagement Survey', date: '2024-07-20' },
  ],
};
