import { KPICard } from '../KPICard';
import { LineChart } from '../charts/LineChart';
import { Activity, Heart, Scale, CheckCircle2, Clock } from 'lucide-react';
import { employeeMockData } from '../../data/mockData';

export const EmployeeDashboard: React.FC = () => {
  const {
    personalWellbeingScore,
    stressLevel,
    moodScore,
    workLifeBalance,
    stressTrend,
    moodTrend,
    recommendations,
    activePrograms,
    upcomingSurveys,
  } = employeeMockData;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">My Wellbeing Dashboard</h1>
        <p className="text-gray-600">Track your personal wellbeing and discover ways to improve</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Personal Wellbeing Score"
          value={personalWellbeingScore}
          change={2}
          trend="up"
          icon={Activity}
          color="primary"
        />
        <KPICard
          title="Stress Level"
          value={stressLevel}
          icon={Heart}
          color="red"
        />
        <KPICard
          title="Mood Score"
          value={moodScore}
          change={0.3}
          trend="up"
          icon={Heart}
          color="green"
        />
        <KPICard
          title="Work-Life Balance"
          value={workLifeBalance}
          icon={Scale}
          color="blue"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Stress Trend */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Stress Level Trend</h2>
          <LineChart
            data={stressTrend}
            dataKeys={[
              { key: 'score', color: '#ef4444', name: 'Stress Level' }
            ]}
            height={250}
          />
        </div>

        {/* Mood Trend */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Mood Trend</h2>
          <LineChart
            data={moodTrend}
            dataKeys={[
              { key: 'score', color: '#10b981', name: 'Mood Score' }
            ]}
            height={250}
          />
        </div>
      </div>

      {/* Recommended Actions & Active Programs Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recommended Wellbeing Actions */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Recommended Actions</h2>
          <div className="space-y-4">
            {recommendations.map((rec) => (
              <div
                key={rec.id}
                className={`flex items-center gap-3 p-4 rounded-lg border ${
                  rec.status === 'active'
                    ? 'bg-green-50 border-green-200'
                    : 'bg-gray-50 border-gray-200'
                }`}
              >
                {rec.status === 'active' ? (
                  <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
                ) : (
                  <Clock className="w-5 h-5 text-gray-400 flex-shrink-0" />
                )}
                <div className="flex-1">
                  <p
                    className={`font-medium ${
                      rec.status === 'active' ? 'text-gray-900' : 'text-gray-600'
                    }`}
                  >
                    {rec.action}
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    {rec.status === 'active' ? 'Active' : 'Pending'}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Active Programs */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Active Programs</h2>
          <div className="space-y-4">
            {activePrograms.map((program) => (
              <div
                key={program.id}
                className={`p-4 rounded-lg border ${
                  program.enrolled
                    ? 'bg-blue-50 border-blue-200'
                    : 'bg-gray-50 border-gray-200'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900">{program.name}</h3>
                    <p className="text-sm text-gray-600 mt-1">{program.description}</p>
                    {program.enrolled && program.startDate && (
                      <p className="text-xs text-gray-500 mt-2">
                        Started: {new Date(program.startDate).toLocaleDateString()}
                      </p>
                    )}
                  </div>
                  <span
                    className={`ml-3 px-3 py-1 rounded-full text-xs font-medium ${
                      program.enrolled
                        ? 'bg-blue-100 text-blue-800'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {program.enrolled ? 'Enrolled' : 'Available'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Upcoming Surveys */}
      {upcomingSurveys && upcomingSurveys.length > 0 && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Upcoming Surveys</h2>
          <div className="space-y-4">
            {upcomingSurveys.map((survey) => (
              <div
                key={survey.id}
                className="flex items-center justify-between p-4 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
              >
                <div>
                  <h3 className="font-semibold text-gray-900">{survey.name}</h3>
                  <p className="text-sm text-gray-600 mt-1">
                    Due: {new Date(survey.date).toLocaleDateString('en-US', {
                      weekday: 'long',
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric',
                    })}
                  </p>
                </div>
                <button className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors text-sm font-medium">
                  View
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
