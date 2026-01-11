import { KPICard } from '../KPICard';
import { DonutChart } from '../charts/DonutChart';
import { LineChart } from '../charts/LineChart';
import { Activity, Users, ClipboardCheck, AlertTriangle } from 'lucide-react';
import { hrMockData } from '../../data/mockData';

export const HRDashboard: React.FC = () => {
  const { companyWellbeingScore, employeesAtRisk, activePrograms, surveyResponseRate, burnoutDistribution, departmentHealth, monthlyTrend, alerts } = hrMockData;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">HR Dashboard</h1>
        <p className="text-gray-600">Company-wide wellbeing insights and analytics</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Company Wellbeing Score"
          value={companyWellbeingScore}
          change={2}
          trend="up"
          icon={Activity}
          color="primary"
        />
        <KPICard
          title="Employees at Risk"
          value={`${employeesAtRisk}%`}
          change={-1}
          trend="down"
          icon={AlertTriangle}
          color="red"
        />
        <KPICard
          title="Active Programs"
          value={activePrograms}
          icon={ClipboardCheck}
          color="blue"
        />
        <KPICard
          title="Survey Response Rate"
          value={`${surveyResponseRate}%`}
          change={3}
          trend="up"
          icon={Users}
          color="green"
        />
      </div>

      {/* Alerts Section */}
      {alerts.length > 0 && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-5 h-5 text-yellow-500" />
            <h2 className="text-xl font-semibold text-gray-900">Alerts</h2>
          </div>
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border-l-4 ${
                  alert.type === 'warning'
                    ? 'bg-yellow-50 border-yellow-400'
                    : alert.type === 'error'
                    ? 'bg-red-50 border-red-400'
                    : 'bg-blue-50 border-blue-400'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900">{alert.title}</h3>
                    <p className="text-sm text-gray-600 mt-1">{alert.message}</p>
                    {alert.department && (
                      <span className="inline-block mt-2 text-xs font-medium text-gray-500 bg-white px-2 py-1 rounded">
                        {alert.department}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Burnout Risk Distribution */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Burnout Risk Distribution</h2>
          <DonutChart data={burnoutDistribution} />
        </div>

        {/* Monthly Wellbeing Trend */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Monthly Wellbeing Trend</h2>
          <LineChart
            data={monthlyTrend}
            dataKeys={[
              { key: 'score', color: '#0ea5e9', name: 'Wellbeing Score' }
            ]}
          />
        </div>
      </div>

      {/* Departmental Health Overview */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Departmental Health Overview</h2>
        <div className="space-y-6">
          {departmentHealth.map((dept) => {
            const riskPercentage = ((dept.atRisk / dept.employees) * 100).toFixed(1);
            const scoreColor =
              dept.score >= 80
                ? 'bg-green-500'
                : dept.score >= 65
                ? 'bg-yellow-500'
                : 'bg-red-500';

            return (
              <div key={dept.name} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="font-semibold text-gray-900 w-32">{dept.name}</span>
                    <span className="text-sm text-gray-600">
                      {dept.employees} employees • {dept.atRisk} at risk ({riskPercentage}%)
                    </span>
                  </div>
                  <span className="text-lg font-bold text-gray-900">{dept.score}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full ${scoreColor} transition-all`}
                    style={{ width: `${dept.score}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
