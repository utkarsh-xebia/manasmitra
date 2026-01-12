import { Router, Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import { User } from '../models/User.js';
import { authenticate, requireRole, AuthRequest } from '../middleware/auth.js';

const router = Router();

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
      status: status || 'active',
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
 * Get list of departments and employee counts (All authenticated roles have read access)
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

    // Check if email is being changed and if it's already taken
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
