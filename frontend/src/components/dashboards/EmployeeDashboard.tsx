import { useState, useEffect } from 'react';
import { KPICard } from '../KPICard';
import { LineChart } from '../charts/LineChart';
import { Activity, Heart, Scale, CheckCircle2, Clock, AlertCircle, Search } from 'lucide-react';
import { employeeMockData } from '../../data/mockData';
import { useSearch } from '../../context/SearchContext';

export const EmployeeDashboard: React.FC = () => {
  const { searchQuery } = useSearch();
  const [latestResult, setLatestResult] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    const fetchLatestResult = async () => {
      try {
        const token = localStorage.getItem('authToken');
        if (!token) return;

        const response = await fetch('http://localhost:5000/api/questionnaire/latest', {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          setLatestResult(data.result);
        }
      } catch (err) {
        console.error('Error fetching latest result:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchLatestResult();
  }, []);

  // Map severity labels to scores for display
  const severityToScore = {
    'Minimal': 90,
    'Mild': 75,
    'Moderate': 55,
    'Moderately Severe': 35,
    'Severe': 15,
  };

  const currentWellbeingScore = latestResult 
    ? (severityToScore[latestResult.severityLabel as keyof typeof severityToScore] || personalWellbeingScore)
    : personalWellbeingScore;

  const currentStressLevel = latestResult ? latestResult.stressLevel : stressLevel;
  const currentMoodScore = latestResult ? latestResult.moodScore : moodScore;
  const currentWorkLifeBalance = latestResult ? latestResult.workLifeBalance : workLifeBalance;
  
  const rawRecommendations = latestResult && latestResult.recommendations?.length > 0 
    ? latestResult.recommendations.map((text: string, index: number) => ({ id: index.toString(), action: text, status: 'active' }))
    : recommendations;

  const filteredRecommendations = rawRecommendations.filter((rec: any) => 
    rec.action.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredSurveys = (upcomingSurveys || []).filter(survey => 
    survey.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const stressChartData = stressTrend.map(item => ({
    name: item.month,
    score: item.score
  }));

  const moodChartData = moodTrend.map(item => ({
    name: item.month,
    score: item.score
  }));

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="mb-6 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">My Wellbeing Dashboard</h1>
          <p className="text-gray-600">Track your personal wellbeing and discover ways to improve</p>
        </div>
        {searchQuery && (
          <div className="flex items-center gap-2 text-sm text-primary-600 bg-primary-50 px-3 py-1.5 rounded-full border border-primary-100 animate-in fade-in slide-in-from-right-2">
            <Search className="w-4 h-4" />
            Filtering by: <strong>"{searchQuery}"</strong>
          </div>
        )}
      </div>

      {latestResult && (
        <div className="bg-primary-50 border border-primary-200 p-4 rounded-xl flex items-center gap-4 mb-6">
          <div className="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center">
            <Activity className="w-6 h-6 text-primary-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">AI Analysis Result: {latestResult.severityLabel}</h3>
            <p className="text-sm text-gray-600">
              Based on your assessment from {new Date(latestResult.createdAt).toLocaleDateString()}, our model predicted this with {Math.round(latestResult.confidenceScore * 100)}% confidence.
            </p>
          </div>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Wellbeing Score"
          value={currentWellbeingScore}
          change={latestResult ? undefined : 2}
          trend={latestResult ? undefined : "up"}
          icon={Activity}
          color="primary"
        />
        <KPICard
          title="Stress Level"
          value={currentStressLevel}
          icon={Heart}
          color="red"
        />
        <KPICard
          title="Mood Score"
          value={currentMoodScore}
          change={latestResult ? undefined : 0.3}
          trend={latestResult ? undefined : "up"}
          icon={Heart}
          color="green"
        />
        <KPICard
          title="Work-Life Balance"
          value={currentWorkLifeBalance}
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
            data={stressChartData}
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
            data={moodChartData}
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
          <h2 className="text-xl font-semibold text-gray-900 mb-6">AI Recommendations</h2>
          <div className="space-y-4">
            {filteredRecommendations.length === 0 ? (
              <p className="text-center text-gray-500 py-4 italic">No recommendations matching your search</p>
            ) : (
              filteredRecommendations.map((rec: any) => (
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
                      {rec.status === 'active' ? 'Recommended for you' : 'Pending'}
                    </p>
                  </div>
                </div>
              ))
            )}
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
            {filteredSurveys.length === 0 ? (
              <p className="text-center text-gray-500 py-4 italic">No surveys matching your search</p>
            ) : (
              filteredSurveys.map((survey) => (
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
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
};
