import { Router, Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { User } from '../models/User.js';
import { authenticate, AuthRequest } from '../middleware/auth.js';

const router = Router();

/**
 * POST /api/auth/register
 * Register a new user
 */
router.post('/register', async (req: Request, res: Response) => {
  try {
    const { name, email, password, role } = req.body;

    // Basic password validation
    if (!password || password.length < 6) {
      res.status(400).json({ error: 'Password must be at least 6 characters long' });
      return;
    }

    // Check if user already exists
    let user = await User.findOne({ email: email.toLowerCase().trim() });
    
    if (user) {
      // If user exists and already has a password AND is not deleted, they are already registered
      if (user.password && user.password.trim() !== '' && !user.isDeleted) {
        res.status(400).json({ error: 'This email is already registered. Please log in instead.' });
        return;
      }
      
      // If user exists but has NO password (HR stub) or was deleted
      // We "claim" or "reactivate" this account
      const salt = await bcrypt.genSalt(10);
      const hashedPassword = await bcrypt.hash(password, salt);
      
      user.password = hashedPassword;
      if (name) user.name = name; 
      
      // Keep HR-assigned role if it exists, otherwise use provided role
      if (!user.role && role) {
        user.role = role;
      }
      
      user.status = 'active';
      user.isDeleted = false;
      user.needsPasswordReset = false; // They are setting it themselves now
      
      await user.save();
    } else {
      // Create completely new user
      const salt = await bcrypt.genSalt(10);
      const hashedPassword = await bcrypt.hash(password, salt);
      
      user = new User({
        name,
        email: email.toLowerCase().trim(),
        password: hashedPassword,
        role: role || 'employee',
      });
      
      await user.save();
    }

    // Generate JWT
    const jwtSecret = process.env.JWT_SECRET || 'your-secret-key-change-in-production';
    const token = jwt.sign(
      { userId: user._id, role: user.role },
      jwtSecret,
      { expiresIn: '7d' }
    );

    res.status(201).json({
      success: true,
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
      },
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Failed to register user' });
  }
});

/**
 * POST /api/auth/login
 * Login a user
 */
router.post('/login', async (req: Request, res: Response) => {
  try {
    const { email, password } = req.body;

    // Find user
    const user = await User.findOne({ email: email.toLowerCase().trim() });
    if (!user || user.isDeleted) {
      res.status(401).json({ error: 'Invalid email or password' });
      return;
    }

    // Check if user has a password (they might be an HR stub)
    if (!user.password || user.password.trim() === '') {
      res.status(401).json({ 
        error: 'Account pending registration. Please use the Sign Up page to set your password.',
        isStub: true 
      });
      return;
    }

    // Check password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      res.status(401).json({ error: 'Invalid email or password' });
      return;
    }

    // Check if password reset is needed
    if (user.needsPasswordReset) {
      const jwtSecret = process.env.JWT_SECRET || 'your-secret-key-change-in-production';
      const token = jwt.sign(
        { userId: user._id, role: user.role, needsPasswordReset: true },
        jwtSecret,
        { expiresIn: '1h' }
      );

      return res.status(200).json({
        success: true,
        message: 'Password reset required',
        needsPasswordReset: true,
        token,
        user: {
          id: user._id,
          name: user.name,
          email: user.email,
          role: user.role,
        },
      });
    }

    // Normal login flow
    const jwtSecret = process.env.JWT_SECRET || 'your-secret-key-change-in-production';
    const token = jwt.sign(
      { userId: user._id, role: user.role },
      jwtSecret,
      { expiresIn: '7d' }
    );

    res.status(200).json({
      success: true,
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
      },
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Failed to login' });
  }
});

/**
 * POST /api/auth/reset-password
 * Reset password (used for force-reset after HR creation)
 */
router.post('/reset-password', authenticate, async (req: AuthRequest, res: Response) => {
  try {
    const { newPassword } = req.body;
    const userId = req.userId;

    if (!newPassword || newPassword.length < 6) {
      res.status(400).json({ error: 'New password must be at least 6 characters long' });
      return;
    }

    const user = await User.findById(userId);
    if (!user) {
      res.status(404).json({ error: 'User not found' });
      return;
    }

    const salt = await bcrypt.genSalt(10);
    user.password = await bcrypt.hash(newPassword, salt);
    user.needsPasswordReset = false;
    
    await user.save();

    res.json({ success: true, message: 'Password reset successfully' });
  } catch (error) {
    console.error('Reset password error:', error);
    res.status(500).json({ error: 'Failed to reset password' });
  }
});

export default router;
