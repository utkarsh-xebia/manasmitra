import React, { useState, useEffect } from 'react';
import { KPICard } from '../KPICard';
import { BarChart } from '../charts/BarChart';
import { LineChart } from '../charts/LineChart';
import { Activity, Users, TrendingUp, AlertTriangle, Search, Loader2 } from 'lucide-react';
import { useSearch } from '../../context/SearchContext';

interface ManagerStats {
  teamWellbeingScore: number;
  wellbeingChange?: number;
  wellbeingTrend?: 'up' | 'down' | 'neutral';
  teamMembersAtRisk: number;
  engagementScore: number;
  engagementChange?: number;
  engagementTrend?: 'up' | 'down' | 'neutral';
  recentSurveyResult: string;
  teamMembers: Array<{
    id: string;
    name: string;
    wellbeingScore: number;
    burnoutRisk: 'low' | 'medium' | 'high';
    engagementScore: number;
  }>;
  workloadVsWellbeing: Array<{
    month: string;
    workload: number;
    score: number;
  }>;
  insights: string[];
}

export const ManagerDashboard: React.FC = () => {
  const { searchQuery } = useSearch();
  const [stats, setStats] = useState<ManagerStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);
        const token = localStorage.getItem('authToken');
        const response = await fetch('http://localhost:5000/api/admin/manager-stats', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          throw new Error('Failed to fetch manager dashboard data');
        }

        const data = await response.json();
        if (data.success) {
          setStats(data.stats);
        } else {
          throw new Error(data.error || 'Failed to fetch manager dashboard data');
        }
      } catch (err: any) {
        console.error('Manager Dashboard fetch error:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
        <p className="text-gray-500 font-medium">Loading team insights...</p>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="p-8 bg-red-50 border border-red-200 rounded-xl text-center">
        <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-red-900 mb-2">Error Loading Dashboard</h2>
        <p className="text-red-700">{error || 'Something went wrong while fetching data.'}</p>
        <button 
          onClick={() => window.location.reload()}
          className="mt-4 px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  const { teamWellbeingScore, wellbeingChange, wellbeingTrend, teamMembersAtRisk, engagementScore, engagementChange, engagementTrend, recentSurveyResult, teamMembers, workloadVsWellbeing, insights } = stats;

  const filteredTeam = teamMembers.filter(member => 
    member.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    member.burnoutRisk.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const teamMemberChartData = filteredTeam.map((member) => ({
    name: member.name.split(' ')[0], // First name only
    score: member.wellbeingScore,
    risk: member.burnoutRisk,
  }));

  const chartData = workloadVsWellbeing.map(item => ({
    name: item.month,
    workload: item.workload,
    score: item.score
  }));

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="mb-6 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Manager Dashboard</h1>
          <p className="text-gray-600">Team wellbeing insights and actionable recommendations</p>
        </div>
        {searchQuery && (
          <div className="flex items-center gap-2 text-sm text-primary-600 bg-primary-50 px-3 py-1.5 rounded-full border border-primary-100 animate-in fade-in slide-in-from-right-2">
            <Search className="w-4 h-4" />
            Filtering by: <strong>"{searchQuery}"</strong>
          </div>
        )}
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Team Wellbeing Score"
          value={teamWellbeingScore}
          change={wellbeingChange}
          trend={wellbeingTrend}
          icon={Activity}
          color="primary"
        />
        <KPICard
          title="Team Members at Risk"
          value={teamMembersAtRisk}
          icon={AlertTriangle}
          color="red"
        />
        <KPICard
          title="Engagement Score"
          value={engagementScore}
          change={engagementChange}
          trend={engagementTrend}
          icon={TrendingUp}
          color="green"
        />
        <KPICard
          title="Recent Survey Result"
          value={recentSurveyResult}
          icon={Users}
          color="blue"
        />
      </div>

      {/* Actionable Insights */}
      {insights && insights.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-5 h-5 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">Actionable Insights</h2>
          </div>
          <ul className="space-y-2">
            {insights.map((insight, index) => (
              <li key={index} className="flex items-start gap-2 text-gray-700">
                <span className="text-blue-600 mt-1">•</span>
                <span>{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Team Member Wellbeing */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Team Member Wellbeing</h2>
          <BarChart
            data={teamMemberChartData}
            dataKeys={[
              { key: 'score', color: '#0ea5e9', name: 'Wellbeing Score' }
            ]}
            height={300}
          />
        </div>

        {/* Workload vs Wellbeing */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Workload vs Wellbeing Trend</h2>
          <LineChart
            data={chartData}
            dataKeys={[
              { key: 'workload', color: '#ef4444', name: 'Workload' },
              { key: 'score', color: '#0ea5e9', name: 'Wellbeing Score' }
            ]}
            height={300}
          />
        </div>
      </div>

      {/* Team Members List */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Team Members</h2>
        <div className="overflow-x-auto">
          {filteredTeam.length === 0 ? (
            <p className="text-center text-gray-500 py-8 italic">No team members matching your search</p>
          ) : (
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Name</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Wellbeing Score</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Burnout Risk</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Engagement</th>
                </tr>
              </thead>
              <tbody>
                {filteredTeam.map((member) => (
                  <tr key={member.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-4 px-4 font-medium text-gray-900">{member.name}</td>
                    <td className="py-4 px-4">
                      <span className="text-gray-900 font-medium">{member.wellbeingScore}</span>
                    </td>
                    <td className="py-4 px-4">
                      <span
                        className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                          member.burnoutRisk === 'low'
                            ? 'bg-green-100 text-green-800'
                            : member.burnoutRisk === 'medium'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-red-100 text-red-800'
                        }`}
                      >
                        {member.burnoutRisk.charAt(0).toUpperCase() + member.burnoutRisk.slice(1)}
                      </span>
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${member.engagementScore}%` }}
                          />
                        </div>
                        <span className="text-sm text-gray-600">{member.engagementScore}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
};
