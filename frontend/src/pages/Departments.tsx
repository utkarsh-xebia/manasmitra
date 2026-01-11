import React, { useState, useEffect } from 'react';
import { Layout } from '../components/Layout';
import { Role } from '../types';
import { Users, Plus, ChevronRight, Building2, UserPlus, Briefcase, User as UserIcon, Shield, Mail, CheckCircle2, AlertCircle } from 'lucide-react';

interface Employee {
  id: string;
  name: string;
  email: string;
  role: string;
  department?: string;
  reportingManager?: { name: string; email: string };
}

interface Department {
  name: string;
  count: number;
  employees: Employee[];
}

export const Departments = () => {
  const [currentRole] = useState<Role>(() => {
    return (localStorage.getItem('userRole') as Role) || 'employee';
  });

  const [departments, setDepartments] = useState<Department[]>([]);
  const [selectedDept, setSelectedDept] = useState<Department | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Form State
  const [showAddModal, setShowAddModal] = useState(false);
  const [managers, setManagers] = useState<{_id: string, name: string}[]>([]);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: 'password123', // Default password
    role: 'employee',
    department: '',
    reportingManager: '',
  });
  const [submitting, setSubmitting] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');

  useEffect(() => {
    fetchDepartments();
    if (currentRole === 'hr') {
      fetchManagers();
    }
  }, [currentRole]);

  const fetchDepartments = async () => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch('http://localhost:5000/api/admin/departments', {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      if (response.ok) {
        setDepartments(data.departments);
      } else {
        setError(data.error || 'Failed to fetch departments');
      }
    } catch (err) {
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  const fetchManagers = async () => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch('http://localhost:5000/api/admin/managers', {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      if (response.ok) {
        setManagers(data.managers);
      }
    } catch (err) {
      console.error('Failed to fetch managers');
    }
  };

  const handleAddEmployee = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setSuccessMsg('');

    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch('http://localhost:5000/api/admin/employees', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();
      if (response.ok) {
        setSuccessMsg('Employee added successfully!');
        setFormData({
          name: '',
          email: '',
          password: 'password123',
          role: 'employee',
          department: '',
          reportingManager: '',
        });
        fetchDepartments();
        setTimeout(() => setShowAddModal(false), 2000);
      } else {
        setError(data.error || 'Failed to add employee');
      }
    } catch (err) {
      setError('Connection error');
    } finally {
      setSubmitting(false);
    }
  };

  if (currentRole !== 'hr') {
    return (
      <Layout currentRole={currentRole}>
        <div className="flex flex-col items-center justify-center min-h-[400px] text-center">
          <Shield className="w-16 h-16 text-gray-300 mb-4" />
          <h2 className="text-2xl font-bold text-gray-900">Access Denied</h2>
          <p className="text-gray-600 mt-2 text-lg">Only HR administrators can access the Departments management module.</p>
        </div>
      </Layout>
    );
  }

  return (
    <Layout currentRole={currentRole}>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Department Management</h1>
            <p className="text-gray-600">Overview of organization structure and employee assignments</p>
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-2 bg-primary-500 text-white px-4 py-2 rounded-lg font-medium hover:bg-primary-600 transition-colors"
          >
            <Plus className="w-5 h-5" />
            Add Employee
          </button>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3 text-red-800">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Departments List */}
          <div className="lg:col-span-1 space-y-4">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              <Building2 className="w-5 h-5 text-primary-500" />
              Departments
            </h2>
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
              {loading ? (
                <div className="p-8 text-center text-gray-500">Loading...</div>
              ) : departments.length === 0 ? (
                <div className="p-8 text-center text-gray-500 italic">No departments found</div>
              ) : (
                <div className="divide-y divide-gray-100">
                  {departments.map((dept) => (
                    <button
                      key={dept.name}
                      onClick={() => setSelectedDept(dept)}
                      className={`w-full flex items-center justify-between p-4 hover:bg-gray-50 transition-colors ${
                        selectedDept?.name === dept.name ? 'bg-primary-50 border-l-4 border-primary-500' : ''
                      }`}
                    >
                      <div className="text-left">
                        <p className={`font-semibold ${selectedDept?.name === dept.name ? 'text-primary-700' : 'text-gray-900'}`}>
                          {dept.name}
                        </p>
                        <p className="text-xs text-gray-500 uppercase">{dept.count} Members</p>
                      </div>
                      <ChevronRight className={`w-5 h-5 ${selectedDept?.name === dept.name ? 'text-primary-500' : 'text-gray-400'}`} />
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Selected Department Employees */}
          <div className="lg:col-span-2 space-y-4">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              <Users className="w-5 h-5 text-primary-500" />
              {selectedDept ? `${selectedDept.name} Members` : 'Select a Department'}
            </h2>
            
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
              {!selectedDept ? (
                <div className="p-12 text-center text-gray-500">
                  <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Users className="w-8 h-8 text-gray-300" />
                  </div>
                  <p>Click on a department to view assigned employees</p>
                </div>
              ) : selectedDept.employees.length === 0 ? (
                <div className="p-12 text-center text-gray-500 italic">No employees assigned to this department</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-gray-50 border-b border-gray-200">
                        <th className="text-left py-3 px-4 font-semibold text-gray-700 text-sm">Name</th>
                        <th className="text-left py-3 px-4 font-semibold text-gray-700 text-sm">Role</th>
                        <th className="text-left py-3 px-4 font-semibold text-gray-700 text-sm">Email</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {selectedDept.employees.map((emp) => (
                        <tr key={emp.id} className="hover:bg-gray-50 transition-colors">
                          <td className="py-4 px-4 font-medium text-gray-900 flex items-center gap-3">
                            <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center text-primary-700 text-xs font-bold">
                              {emp.name.charAt(0)}
                            </div>
                            {emp.name}
                          </td>
                          <td className="py-4 px-4">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${
                              emp.role === 'hr' ? 'bg-purple-100 text-hr-800' :
                              emp.role === 'manager' ? 'bg-blue-100 text-blue-800' :
                              'bg-green-100 text-green-800'
                            }`}>
                              {emp.role}
                            </span>
                          </td>
                          <td className="py-4 px-4 text-sm text-gray-600">{emp.email}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Add Employee Modal */}
        {showAddModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-xl w-full max-w-lg overflow-hidden animate-in fade-in zoom-in duration-200">
              <div className="p-6 border-b border-gray-100 flex items-center justify-between bg-primary-50">
                <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                  <UserPlus className="w-6 h-6 text-primary-600" />
                  Add New Employee
                </h3>
                <button onClick={() => setShowAddModal(false)} className="text-gray-400 hover:text-gray-600 transition-colors">
                  <Plus className="w-6 h-6 transform rotate-45" />
                </button>
              </div>
              
              <form onSubmit={handleAddEmployee} className="p-6 space-y-4">
                {successMsg && (
                  <div className="p-3 bg-green-50 border border-green-200 text-green-700 rounded-lg flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5" />
                    {successMsg}
                  </div>
                )}

                <div className="grid grid-cols-1 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center gap-1">
                      <UserIcon className="w-4 h-4" /> Full Name
                    </label>
                    <input
                      required
                      type="text"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500"
                      placeholder="Jane Smith"
                      value={formData.name}
                      onChange={(e) => setFormData({...formData, name: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center gap-1">
                      <Mail className="w-4 h-4" /> Email Address
                    </label>
                    <input
                      required
                      type="email"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500"
                      placeholder="jane@company.com"
                      value={formData.email}
                      onChange={(e) => setFormData({...formData, email: e.target.value})}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center gap-1">
                        <Briefcase className="w-4 h-4" /> Department
                      </label>
                      <input
                        required
                        type="text"
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500"
                        placeholder="e.g. Sales"
                        value={formData.department}
                        onChange={(e) => setFormData({...formData, department: e.target.value})}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center gap-1">
                        <Shield className="w-4 h-4" /> System Role
                      </label>
                      <select
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500"
                        value={formData.role}
                        onChange={(e) => setFormData({...formData, role: e.target.value})}
                      >
                        <option value="employee">Employee</option>
                        <option value="manager">Manager</option>
                        <option value="hr">HR Admin</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Reporting Manager (Optional)</label>
                    <select
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500"
                      value={formData.reportingManager}
                      onChange={(e) => setFormData({...formData, reportingManager: e.target.value})}
                    >
                      <option value="">Select a Manager</option>
                      {managers.map(m => (
                        <option key={m._id} value={m._id}>{m.name}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="pt-4 flex gap-3">
                  <button
                    type="button"
                    onClick={() => setShowAddModal(false)}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    disabled={submitting}
                    type="submit"
                    className="flex-1 px-4 py-2 bg-primary-500 text-white rounded-lg font-medium hover:bg-primary-600 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {submitting ? 'Adding...' : 'Add Employee'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
};
