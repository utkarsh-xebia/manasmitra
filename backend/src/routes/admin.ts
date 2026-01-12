import { Router, Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import { User } from '../models/User.js';
import { MentalHealthResult } from '../models/MentalHealthResult.js';
import { authenticate, requireRole, AuthRequest } from '../middleware/auth.js';
import mongoose from 'mongoose';

const router = Router();

/**
 * GET /api/admin/hr-stats
 * Get aggregated wellbeing stats for the entire company (HR only)
 */
router.get('/hr-stats', authenticate, requireRole(['hr']), async (req: Request, res: Response) => {
  try {
    // 1. Get latest result for each user
    const latestResults = await MentalHealthResult.aggregate([
      { $sort: { createdAt: -1 } },
      {
        $group: {
          _id: "$userId",
          latestResult: { $first: "$$ROOT" }
        }
      },
      {
        $lookup: {
          from: "users",
          localField: "_id",
          foreignField: "_id",
          as: "user"
        }
      },
      { $unwind: "$user" },
      { $match: { "user.isDeleted": { $ne: true } } }
    ]);

    if (latestResults.length === 0) {
      return res.json({
        success: true,
        stats: {
          companyWellbeingScore: 0,
          employeesAtRisk: 0,
          activePrograms: 0,
          surveyResponseRate: 0,
          burnoutDistribution: [
            { name: 'Low Risk', value: 0, color: '#10b981' },
            { name: 'Medium Risk', value: 0, color: '#f59e0b' },
            { name: 'High Risk', value: 0, color: '#ef4444' }
          ],
          departmentHealth: [],
          monthlyTrend: [],
          alerts: []
        }
      });
    }

    // 2. Calculate KPI metrics
    const totalMoodScore = latestResults.reduce((sum, r) => sum + r.latestResult.moodScore, 0);
    const companyWellbeingScore = Math.round((totalMoodScore / latestResults.length) * 10);

    const atRiskCount = latestResults.filter(r => 
      ['Moderate', 'Moderately Severe', 'Severe'].includes(r.latestResult.severityLabel)
    ).length;
    const employeesAtRisk = Math.round((atRiskCount / latestResults.length) * 100);

    // Calculate changes (comparing with results from > 30 days ago)
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const previousResults = await MentalHealthResult.aggregate([
      { $match: { createdAt: { $lt: thirtyDaysAgo } } },
      { $sort: { createdAt: -1 } },
      {
        $group: {
          _id: "$userId",
          latestResult: { $first: "$$ROOT" }
        }
      }
    ]);

    let wellbeingChange = 0;
    let wellbeingTrend: 'up' | 'down' | 'neutral' = 'neutral';
    
    if (previousResults.length > 0) {
      const previousTotalMood = previousResults.reduce((sum, r) => sum + r.latestResult.moodScore, 0);
      const previousScore = Math.round((previousTotalMood / previousResults.length) * 10);
      wellbeingChange = companyWellbeingScore - previousScore;
      wellbeingTrend = wellbeingChange > 0 ? 'up' : (wellbeingChange < 0 ? 'down' : 'neutral');
    }

    let riskChange = 0;
    let riskTrend: 'up' | 'down' | 'neutral' = 'neutral';
    if (previousResults.length > 0) {
      const prevAtRisk = previousResults.filter(r => 
        ['Moderate', 'Moderately Severe', 'Severe'].includes(r.latestResult.severityLabel)
      ).length;
      const prevRiskPercent = Math.round((prevAtRisk / previousResults.length) * 100);
      riskChange = employeesAtRisk - prevRiskPercent;
      riskTrend = riskChange > 0 ? 'up' : (riskChange < 0 ? 'down' : 'neutral');
    }

    // 3. Burnout Distribution
    const distribution = {
      'Low Risk': 0,
      'Medium Risk': 0,
      'High Risk': 0
    };

    latestResults.forEach(r => {
      const label = r.latestResult.severityLabel;
      if (['Minimal', 'Mild'].includes(label)) distribution['Low Risk']++;
      else if (label === 'Moderate') distribution['Medium Risk']++;
      else distribution['High Risk']++;
    });

    const burnoutDistribution = [
      { name: 'Low Risk', value: distribution['Low Risk'], color: '#10b981' },
      { name: 'Medium Risk', value: distribution['Medium Risk'], color: '#f59e0b' },
      { name: 'High Risk', value: distribution['High Risk'], color: '#ef4444' }
    ];

    // 4. Departmental Health
    const deptStats: Record<string, { totalScore: number, count: number, atRisk: number, employees: number }> = {};
    
    const allEmployees = await User.find({ isDeleted: { $ne: true } });
    allEmployees.forEach(emp => {
      const dept = emp.department || 'Unassigned';
      if (!deptStats[dept]) {
        deptStats[dept] = { totalScore: 0, count: 0, atRisk: 0, employees: 0 };
      }
      deptStats[dept].employees++;
    });

    latestResults.forEach(r => {
      const dept = r.user.department || 'Unassigned';
      if (deptStats[dept]) {
        deptStats[dept].totalScore += r.latestResult.moodScore;
        deptStats[dept].count++;
        if (['Moderate', 'Moderately Severe', 'Severe'].includes(r.latestResult.severityLabel)) {
          deptStats[dept].atRisk++;
        }
      }
    });

    const departmentHealth = Object.entries(deptStats).map(([name, data]) => ({
      name,
      score: data.count > 0 ? Math.round((data.totalScore / data.count) * 10) : 0,
      employees: data.employees,
      atRisk: data.atRisk
    })).sort((a, b) => b.score - a.score);

    // 5. Monthly Trend
    const monthlyTrendResults = await MentalHealthResult.aggregate([
      {
        $group: {
          _id: {
            year: { $year: "$createdAt" },
            month: { $month: "$createdAt" }
          },
          avgScore: { $avg: "$moodScore" }
        }
      },
      { $sort: { "_id.year": 1, "_id.month": 1 } },
      { $limit: 6 }
    ]);

    const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const monthlyTrend = monthlyTrendResults.map(r => ({
      month: monthNames[r._id.month - 1],
      score: Math.round(r.avgScore * 10)
    }));

    // 6. Survey Response Rate
    const totalActiveEmployees = await User.countDocuments({ isDeleted: { $ne: true } });
    const surveyResponseRate = totalActiveEmployees > 0 
      ? Math.round((latestResults.length / totalActiveEmployees) * 100) 
      : 0;

    let responseRateChange = 0;
    let responseRateTrend: 'up' | 'down' | 'neutral' = 'neutral';
    if (previousResults.length > 0 && totalActiveEmployees > 0) {
      const previousResponseRate = Math.round((previousResults.length / totalActiveEmployees) * 100);
      responseRateChange = surveyResponseRate - previousResponseRate;
      responseRateTrend = responseRateChange > 0 ? 'up' : (responseRateChange < 0 ? 'down' : 'neutral');
    }

    // 7. Alerts
    const alerts = [];
    if (employeesAtRisk > 20) {
      alerts.push({
        id: 'alert-1',
        type: 'error',
        title: 'High Burnout Risk',
        message: `Over ${employeesAtRisk}% of employees are showing signs of moderate to severe stress.`
      });
    }

    departmentHealth.filter(d => d.score < 60).forEach((d, i) => {
      alerts.push({
        id: `alert-dept-${i}`,
        type: 'warning',
        title: 'Low Wellbeing Score',
        message: `The ${d.name} department has a wellbeing score of ${d.score}. Consider a team check-in.`,
        department: d.name
      });
    });

    const activeProgramsCount = Object.keys(deptStats).filter(d => deptStats[d].count > 0).length + 2;

    res.json({
      success: true,
      stats: {
        companyWellbeingScore,
        wellbeingChange,
        wellbeingTrend,
        employeesAtRisk,
        riskChange,
        riskTrend,
        activePrograms: activeProgramsCount,
        surveyResponseRate,
        responseRateChange,
        responseRateTrend,
        burnoutDistribution,
        departmentHealth,
        monthlyTrend,
        alerts
      }
    });

  } catch (error) {
    console.error('Error fetching HR stats:', error);
    res.status(500).json({ error: 'Failed to fetch HR statistics' });
  }
});

/**
 * GET /api/admin/manager-stats
 * Get aggregated wellbeing stats for a manager's team
 */
router.get('/manager-stats', authenticate, requireRole(['manager']), async (req: AuthRequest, res: Response) => {
  try {
    const managerId = req.userId;

    const teamMembers = await User.find({ 
      reportingManager: managerId,
      isDeleted: { $ne: true }
    });

    if (teamMembers.length === 0) {
      return res.json({
        success: true,
        stats: {
          teamWellbeingScore: 0,
          teamMembersAtRisk: 0,
          engagementScore: 0,
          recentSurveyResult: '0%',
          teamMembers: [],
          workloadVsWellbeing: [],
          insights: ['Your team currently has no members assigned or active.']
        }
      });
    }

    const teamUserIds = teamMembers.map(m => m._id);

    const teamResults = await MentalHealthResult.aggregate([
      { $match: { userId: { $in: teamUserIds } } },
      { $sort: { createdAt: -1 } },
      {
        $group: {
          _id: "$userId",
          latestResult: { $first: "$$ROOT" }
        }
      }
    ]);

    const totalMoodScore = teamResults.reduce((sum, r) => sum + r.latestResult.moodScore, 0);
    const teamWellbeingScore = teamResults.length > 0 
      ? Math.round((totalMoodScore / teamResults.length) * 10) 
      : 0;

    const atRiskCount = teamResults.filter(r => 
      ['Moderate', 'Moderately Severe', 'Severe'].includes(r.latestResult.severityLabel)
    ).length;

    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const previousTeamResults = await MentalHealthResult.aggregate([
      { $match: { userId: { $in: teamUserIds }, createdAt: { $lt: thirtyDaysAgo } } },
      { $sort: { createdAt: -1 } },
      {
        $group: {
          _id: "$userId",
          latestResult: { $first: "$$ROOT" }
        }
      }
    ]);

    let wellbeingChange = 0;
    let wellbeingTrend: 'up' | 'down' | 'neutral' = 'neutral';
    
    if (previousTeamResults.length > 0) {
      const previousTotalMood = previousTeamResults.reduce((sum, r) => sum + r.latestResult.moodScore, 0);
      const previousScore = Math.round((previousTotalMood / previousTeamResults.length) * 10);
      wellbeingChange = teamWellbeingScore - previousScore;
      wellbeingTrend = wellbeingChange > 0 ? 'up' : (wellbeingChange < 0 ? 'down' : 'neutral');
    }

    const teamMembersData = teamMembers.map(member => {
      const result = teamResults.find(r => r._id.toString() === member._id.toString());
      const score = result ? Math.round(result.latestResult.moodScore * 10) : 0;
      return {
        id: member._id,
        name: member.name,
        email: member.email,
        role: member.role,
        wellbeingScore: score,
        burnoutRisk: result ? (
          ['Minimal', 'Mild'].includes(result.latestResult.severityLabel) ? 'low' :
          result.latestResult.severityLabel === 'Moderate' ? 'medium' : 'high'
        ) : 'low',
        engagementScore: result ? Math.min(100, 75 + (score / 10) * 2) : 45 
      };
    });

    const avgEngagement = teamMembersData.length > 0
      ? Math.round(teamMembersData.reduce((sum, m) => sum + m.engagementScore, 0) / teamMembersData.length)
      : 0;

    const monthlyTrendResults = await MentalHealthResult.aggregate([
      { $match: { userId: { $in: teamUserIds } } },
      {
        $group: {
          _id: {
            year: { $year: "$createdAt" },
            month: { $month: "$createdAt" }
          },
          avgScore: { $avg: "$moodScore" },
          avgStress: { $avg: "$stressLevel" }
        }
      },
      { $sort: { "_id.year": 1, "_id.month": 1 } },
      { $limit: 6 }
    ]);

    const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const workloadVsWellbeing = monthlyTrendResults.map(r => ({
      month: monthNames[r._id.month - 1],
      workload: Math.round(r.avgStress * 10),
      score: Math.round(r.avgScore * 10)
    }));

    const insights = [];
    if (atRiskCount > 0) {
      insights.push(`${atRiskCount} team member(s) are at high risk of burnout. Recommend a 1-on-1 session.`);
    }
    if (teamWellbeingScore < 70 && teamWellbeingScore > 0) {
      insights.push(`Overall team wellbeing is lower than average (${teamWellbeingScore}). Consider reducing workload.`);
    } else if (teamWellbeingScore >= 70) {
      insights.push("Team wellbeing is stable. Continue regular check-ins.");
    } else {
      insights.push("Waiting for first team assessments to generate insights.");
    }

    res.json({
      success: true,
      stats: {
        teamWellbeingScore,
        wellbeingChange,
        wellbeingTrend,
        teamMembersAtRisk: atRiskCount,
        engagementScore: avgEngagement,
        engagementChange: teamResults.length > 0 ? 3 : 0, 
        engagementTrend: 'up' as const,
        recentSurveyResult: `${Math.round((teamResults.length / teamMembers.length) * 100)}%`,
        teamMembers: teamMembersData,
        workloadVsWellbeing,
        insights
      }
    });

  } catch (error) {
    console.error('Error fetching manager stats:', error);
    res.status(500).json({ error: 'Failed to fetch manager statistics' });
  }
});

/**
 * GET /api/admin/employees
 * Get all employees (All authenticated roles have read access)
 */
router.get('/employees', authenticate, requireRole(['hr', 'manager', 'employee']), async (req: Request, res: Response) => {
  try {
    const employees = await User.find({ isDeleted: { $ne: true } }, '-password')
      .populate('reportingManager', 'name email')
      .sort({ createdAt: -1 });
    res.json({ success: true, employees });
  } catch (error) {
    console.error('Error fetching employees:', error);
    res.status(500).json({ error: 'Failed to fetch employees' });
  }
});

/**
 * GET /api/admin/managers
 * Get all users with manager role (HR only)
 */
router.get('/managers', authenticate, requireRole(['hr']), async (req: Request, res: Response) => {
  try {
    const managers = await User.find({ role: 'manager' }, 'name email department')
      .sort({ name: 1 });
    res.json({ success: true, managers });
  } catch (error) {
    console.error('Error fetching managers:', error);
    res.status(500).json({ error: 'Failed to fetch managers' });
  }
});

/**
 * POST /api/admin/employees
 * Create a new employee (HR only)
 */
router.post('/employees', authenticate, requireRole(['hr']), async (req: Request, res: Response) => {
  try {
    const { name, email, password, role, department, reportingManager, status } = req.body;

    // Check if user already exists
    let user = await User.findOne({ email: email.toLowerCase().trim() });
    
    if (user) {
      // If user exists, we update them instead of failing
      user.isDeleted = false;
      user.status = status || user.status || 'active';
      if (name) user.name = name;
      if (role) user.role = role;
      if (department) user.department = department;
      if (reportingManager) user.reportingManager = reportingManager;
      
      if (password) {
        const salt = await bcrypt.genSalt(10);
        user.password = await bcrypt.hash(password, salt);
        user.needsPasswordReset = true;
      }
      
      await user.save();
      
      res.status(200).json({
        success: true,
        message: user.password ? 'Employee profile updated and password set' : 'Employee profile updated',
        user: {
          id: user._id,
          name: user.name,
          email: user.email,
          role: user.role,
          department: user.department,
        },
      });
      return;
    }

    let hashedPassword = undefined;
    if (password) {
      const salt = await bcrypt.genSalt(10);
      hashedPassword = await bcrypt.hash(password, salt);
    }

    user = new User({
      name,
      email: email.toLowerCase().trim(),
      password: hashedPassword,
      role: role || 'employee',
      department,
      reportingManager: reportingManager || undefined,
      status: status || 'active',
      needsPasswordReset: !!password,
    });

    await user.save();

    res.status(201).json({
      success: true,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        department: user.department,
      },
    });
  } catch (error) {
    console.error('Error creating employee:', error);
    res.status(500).json({ error: 'Failed to create employee' });
  }
});

/**
 * GET /api/admin/departments
 * Get list of departments and employee counts
 */
router.get('/departments', authenticate, requireRole(['hr', 'manager', 'employee']), async (req: Request, res: Response) => {
  try {
    const departments = await User.aggregate([
      { $match: { isDeleted: { $ne: true } } },
      {
        $group: {
          _id: "$department",
          count: { $sum: 1 },
          employees: { 
            $push: { 
              id: "$_id", 
              name: "$name", 
              email: "$email", 
              role: "$role",
              status: "$status"
            } 
          }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    const formattedDepartments = departments.map(d => ({
      name: d._id || 'Unassigned',
      count: d.count,
      employees: d.employees
    }));

    res.json({ success: true, departments: formattedDepartments });
  } catch (error) {
    console.error('Error fetching departments:', error);
    res.status(500).json({ error: 'Failed to fetch departments' });
  }
});

/**
 * PATCH /api/admin/employees/:id
 * Update employee details (HR only)
 */
router.patch('/employees/:id', authenticate, requireRole(['hr']), async (req: Request, res: Response) => {
  try {
    const { name, email, role, department, reportingManager, status } = req.body;
    const employeeId = req.params.id;

    const employee = await User.findById(employeeId);
    if (!employee || employee.isDeleted) {
      res.status(404).json({ error: 'Employee not found' });
      return;
    }

    if (email && email !== employee.email) {
      const emailExists = await User.findOne({ email, _id: { $ne: employeeId } });
      if (emailExists) {
        res.status(400).json({ error: 'Email already in use' });
        return;
      }
    }

    const updatedEmployee = await User.findByIdAndUpdate(
      employeeId,
      {
        $set: {
          name: name || employee.name,
          email: email || employee.email,
          role: role || employee.role,
          department: department || employee.department,
          reportingManager: reportingManager === "" ? undefined : (reportingManager || employee.reportingManager),
          status: status || employee.status,
        }
      },
      { new: true, select: '-password' }
    ).populate('reportingManager', 'name email');

    res.json({ success: true, user: updatedEmployee });
  } catch (error) {
    console.error('Error updating employee:', error);
    res.status(500).json({ error: 'Failed to update employee' });
  }
});

/**
 * DELETE /api/admin/employees/:id
 * Soft delete employee (HR only)
 */
router.delete('/employees/:id', authenticate, requireRole(['hr']), async (req: Request, res: Response) => {
  try {
    const employeeId = req.params.id;

    const employee = await User.findById(employeeId);
    if (!employee || employee.isDeleted) {
      res.status(404).json({ error: 'Employee not found' });
      return;
    }

    employee.isDeleted = true;
    employee.status = 'inactive';
    await employee.save();

    res.json({ success: true, message: 'Employee deleted successfully (soft delete)' });
  } catch (error) {
    console.error('Error deleting employee:', error);
    res.status(500).json({ error: 'Failed to delete employee' });
  }
});

export default router;
