import { Router, Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import { User } from '../models/User.js';
import { authenticate, requireRole, AuthRequest } from '../middleware/auth.js';

const router = Router();

/**
 * GET /api/admin/employees
 * Get all employees (HR only)
 */
router.get('/employees', authenticate, requireRole(['hr']), async (req: Request, res: Response) => {
  try {
    const employees = await User.find({}, '-password')
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
    const { name, email, password, role, department, reportingManager } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      res.status(400).json({ error: 'Employee with this email already exists' });
      return;
    }

    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // Create user
    const user = new User({
      name,
      email,
      password: hashedPassword,
      role,
      department,
      reportingManager: reportingManager || undefined,
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
 * Get list of departments and employee counts (HR only)
 */
router.get('/departments', authenticate, requireRole(['hr']), async (req: Request, res: Response) => {
  try {
    const departments = await User.aggregate([
      {
        $group: {
          _id: "$department",
          count: { $sum: 1 },
          employees: { 
            $push: { 
              id: "$_id", 
              name: "$name", 
              email: "$email", 
              role: "$role" 
            } 
          }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    // Clean up null/empty department names
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

export default router;
