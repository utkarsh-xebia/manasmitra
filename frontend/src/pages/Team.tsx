import React, { useState, useEffect } from 'react';
import { Layout } from '../components/Layout';
import { Role } from '../types';
import { 
  Users, ChevronRight, User as UserIcon, Shield, Mail, 
  AlertCircle, Loader2
} from 'lucide-react';

interface Employee {
  id: string;
  name: string;
  email: string;
  role: string;
  wellbeingScore: number;
  burnoutRisk: 'low' | 'medium' | 'high';
  engagementScore: number;
}

export const Team = () => {
  const [currentRole] = useState<Role>(() => {
    return (localStorage.getItem('userRole') as Role) || 'manager';
  });

  const [teamMembers, setTeamMembers] = useState<Employee[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTeamData();
  }, []);

  const fetchTeamData = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('authToken');
      const response = await fetch('http://localhost:5000/api/admin/manager-stats', {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      
      if (response.ok && data.success) {
        setTeamMembers(data.stats.teamMembers);
      } else {
        setError(data.error || 'Failed to fetch team data');
      }
    } catch (err) {
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout currentRole={currentRole}>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">My Team</h1>
            <p className="text-gray-600">Monitor and support your team's wellbeing and performance</p>
          </div>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3 text-red-800">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          {loading ? (
            <div className="flex flex-col items-center justify-center p-12 space-y-4">
              <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
              <p className="text-gray-500 font-medium">Loading team details...</p>
            </div>
          ) : teamMembers.length === 0 ? (
            <div className="p-12 text-center text-gray-500 italic">
              <Users className="w-12 h-12 text-gray-300 mx-auto mb-4" />
              <p>No team members found under your management</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-gray-50 border-b border-gray-200">
                    <th className="text-left py-3 px-6 font-semibold text-gray-700 text-sm uppercase tracking-wider">Name</th>
                    <th className="text-left py-3 px-6 font-semibold text-gray-700 text-sm uppercase tracking-wider">Wellbeing Score</th>
                    <th className="text-left py-3 px-6 font-semibold text-gray-700 text-sm uppercase tracking-wider">Burnout Risk</th>
                    <th className="text-left py-3 px-6 font-semibold text-gray-700 text-sm uppercase tracking-wider">Engagement</th>
                    <th className="text-left py-3 px-6 font-semibold text-gray-700 text-sm uppercase tracking-wider">Email</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {teamMembers.map((emp) => (
                    <tr key={emp.id} className="hover:bg-gray-50 transition-colors">
                      <td className="py-4 px-6 font-medium flex items-center gap-3 text-gray-900">
                        <div className="w-10 h-10 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-sm font-bold shadow-sm">
                          {emp.name.charAt(0)}
                        </div>
                        <div>
                          <p className="font-semibold">{emp.name}</p>
                          <p className="text-xs text-gray-500 capitalize">{emp.role}</p>
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        <div className="flex items-center gap-2">
                          <span className="text-lg font-bold text-gray-900">{emp.wellbeingScore}</span>
                          <div className="w-16 bg-gray-100 rounded-full h-1.5 overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${
                                emp.wellbeingScore >= 80 ? 'bg-green-500' :
                                emp.wellbeingScore >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                              }`}
                              style={{ width: `${emp.wellbeingScore}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold capitalize ${
                          emp.burnoutRisk === 'low' ? 'bg-green-100 text-green-800 border border-green-200' :
                          emp.burnoutRisk === 'medium' ? 'bg-yellow-100 text-yellow-800 border border-yellow-200' :
                          'bg-red-100 text-red-800 border border-red-200'
                        }`}>
                          {emp.burnoutRisk} Risk
                        </span>
                      </td>
                      <td className="py-4 px-6">
                        <div className="flex items-center gap-2">
                          <div className="w-24 bg-gray-100 rounded-full h-2 shadow-inner overflow-hidden">
                            <div
                              className="bg-blue-500 h-full rounded-full transition-all duration-500"
                              style={{ width: `${emp.engagementScore}%` }}
                            />
                          </div>
                          <span className="text-sm font-medium text-gray-600">{emp.engagementScore}%</span>
                        </div>
                      </td>
                      <td className="py-4 px-6 text-sm text-gray-600">
                        <div className="flex items-center gap-2">
                          <Mail className="w-4 h-4 text-gray-400" />
                          {emp.email}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
};
