import React from 'react';
import { KPICard } from '../KPICard';
import { DonutChart } from '../charts/DonutChart';
import { LineChart } from '../charts/LineChart';
import {
  Activity,
  Users,
  ClipboardCheck,
  AlertTriangle,
  Search,
} from 'lucide-react';
import { hrMockData } from '../../data/mockData';
import { useSearch } from '../../context/SearchContext';

export const HRDashboard: React.FC = () => {
  const { searchQuery } = useSearch();

  const {
    companyWellbeingScore,
    employeesAtRisk,
    activePrograms,
    surveyResponseRate,
    burnoutDistribution,
    departmentHealth,
    monthlyTrend,
    alerts,
  } = hrMockData;

  const filteredDepts = departmentHealth.filter((dept) =>
    dept.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredAlerts = alerts.filter(
    (alert) =>
      alert.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      alert.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      alert.department?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const chartData = monthlyTrend.map(item => ({
    name: item.month,
    score: item.score
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            HR Dashboard
          </h1>
          <p className="text-gray-600">
            Company-wide wellbeing insights and analytics
          </p>
        </div>

        {searchQuery && (
          <div className="flex items-center gap-2 text-sm text-primary-600 bg-primary-50 px-3 py-1.5 rounded-full border border-primary-100">
            <Search className="w-4 h-4" />
            Filtering by <strong>"{searchQuery}"</strong>
          </div>
        )}
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

      {/* Alerts */}
      {filteredAlerts.length > 0 && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-5 h-5 text-yellow-500" />
            <h2 className="text-xl font-semibold text-gray-900">
              Alerts
            </h2>
          </div>

          <div className="space-y-3">
            {filteredAlerts.map((alert) => (
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
                <h3 className="font-semibold text-gray-900">
                  {alert.title}
                </h3>
                <p className="text-sm text-gray-600 mt-1">
                  {alert.message}
                </p>

                {alert.department && (
                  <span className="inline-block mt-2 text-xs font-medium text-gray-500 bg-white px-2 py-1 rounded">
                    {alert.department}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">
            Burnout Risk Distribution
          </h2>
          <DonutChart data={burnoutDistribution} />
        </div>

        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">
            Monthly Wellbeing Trend
          </h2>
          <LineChart
            data={chartData}
            dataKeys={[
              { key: 'score', color: '#0ea5e9', name: 'Wellbeing Score' },
            ]}
          />
        </div>
      </div>

      {/* Department Health */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">
          Departmental Health Overview
        </h2>

        {filteredDepts.length === 0 ? (
          <p className="text-center text-gray-500 py-8 italic">
            No departments matching your search
          </p>
        ) : (
          <div className="space-y-6">
            {filteredDepts.map((dept) => {
              const riskPercentage = (
                (dept.atRisk / dept.employees) *
                100
              ).toFixed(1);

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
                      <span className="font-semibold text-gray-900 w-32">
                        {dept.name}
                      </span>
                      <span className="text-sm text-gray-600">
                        {dept.employees} employees • {dept.atRisk} at risk (
                        {riskPercentage}%)
                      </span>
                    </div>
                    <span className="text-lg font-bold text-gray-900">
                      {dept.score}
                    </span>
                  </div>

                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full ${scoreColor}`}
                      style={{ width: `${dept.score}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
