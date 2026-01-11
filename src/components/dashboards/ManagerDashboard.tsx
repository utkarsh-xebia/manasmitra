import { KPICard } from '../KPICard';
import { BarChart } from '../charts/BarChart';
import { LineChart } from '../charts/LineChart';
import { Activity, Users, TrendingUp, AlertTriangle } from 'lucide-react';
import { managerMockData } from '../../data/mockData';

export const ManagerDashboard: React.FC = () => {
  const { teamWellbeingScore, teamMembersAtRisk, engagementScore, recentSurveyResult, teamMembers, workloadVsWellbeing, insights } = managerMockData;

  const riskColors: Record<string, string> = {
    low: '#10b981',
    medium: '#f59e0b',
    high: '#ef4444',
  };

  const teamMemberChartData = teamMembers.map((member) => ({
    name: member.name.split(' ')[0], // First name only
    score: member.wellbeingScore,
    risk: member.burnoutRisk,
  }));

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Manager Dashboard</h1>
        <p className="text-gray-600">Team wellbeing insights and actionable recommendations</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Team Wellbeing Score"
          value={teamWellbeingScore}
          change={1}
          trend="up"
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
          change={2}
          trend="up"
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
            data={workloadVsWellbeing}
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
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Team Members at Risk</h2>
        <div className="overflow-x-auto">
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
              {teamMembers.map((member) => (
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
        </div>
      </div>
    </div>
  );
};
