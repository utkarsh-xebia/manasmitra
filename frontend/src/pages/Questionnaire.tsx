import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle2, AlertCircle } from 'lucide-react';

// PHQ-9 standard questions
const PHQ9_QUESTIONS = [
  'Little interest or pleasure in doing things',
  'Feeling down, depressed, or hopeless',
  'Trouble falling or staying asleep, or sleeping too much',
  'Feeling tired or having little energy',
  'Poor appetite or overeating',
  'Feeling bad about yourself—or that you are a failure or have let yourself or your family down',
  'Trouble concentrating on things, such as reading the newspaper or watching television',
  'Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual',
  'Thoughts that you would be better off dead or of hurting yourself in some way',
];

const OPTIONS = [
  { label: 'Not at all', value: 0 },
  { label: 'Several days', value: 1 },
  { label: 'More than half the days', value: 2 },
  { label: 'Nearly every day', value: 3 },
];

interface QuestionnaireState {
  answers: (number | null)[];
  age: string;
  gender: string;
  sleepQuality: string;
  studyPressure: string;
  financialPressure: string;
  isSubmitting: boolean;
  error: string | null;
}

export const Questionnaire = () => {
  const navigate = useNavigate();
  const [state, setState] = useState<QuestionnaireState>({
    answers: new Array(9).fill(null),
    age: '25',
    gender: 'Male',
    sleepQuality: 'Average',
    studyPressure: 'Average',
    financialPressure: 'Average',
    isSubmitting: false,
    error: null,
  });

  const handleAnswerChange = (questionIndex: number, value: number) => {
    const newAnswers = [...state.answers];
    newAnswers[questionIndex] = value;
    setState({ ...state, answers: newAnswers, error: null });
  };

  const calculateTotalScore = (): number => {
    return state.answers.reduce((sum, val) => (sum as number) + (val || 0), 0) as number;
  };

  const isComplete = (): boolean => {
    return state.answers.every((answer) => answer !== null) && !!state.age && !!state.gender;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!isComplete()) {
      setState({ ...state, error: 'Please answer all questions and fill in profile details' });
      return;
    }

    setState({ ...state, isSubmitting: true, error: null });

    try {
      const token = localStorage.getItem('authToken');
      if (!token) {
        navigate('/login');
        return;
      }

      const response = await fetch('http://localhost:5000/api/questionnaire/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          answers: state.answers,
          age: parseInt(state.age),
          gender: state.gender,
          sleepQuality: state.sleepQuality,
          studyPressure: state.studyPressure,
          financialPressure: state.financialPressure,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to submit questionnaire');
      }

      const data = await response.json();
      // Redirect to dashboard with success message
      navigate('/dashboard', { state: { questionnaireResult: data.result } });
    } catch (error) {
      console.error('Questionnaire submission error:', error);
      setState({
        ...state,
        isSubmitting: false,
        error: error instanceof Error ? error.message : 'Failed to submit questionnaire',
      });
    }
  };

  const totalScore = calculateTotalScore();

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">PHQ-9 Mental Health Assessment</h1>
          <p className="text-gray-600">
            Over the last 2 weeks, how often have you been bothered by any of the following problems?
          </p>
        </div>

        {/* Error Message */}
        {state.error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
            <p className="text-red-800">{state.error}</p>
          </div>
        )}

        {/* Profile Details */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Profile Details</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Age</label>
              <input
                type="number"
                value={state.age}
                onChange={(e) => setState({ ...state, age: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
                placeholder="e.g. 25"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Gender</label>
              <select
                value={state.gender}
                onChange={(e) => setState({ ...state, gender: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Sleep Quality</label>
              <select
                value={state.sleepQuality}
                onChange={(e) => setState({ ...state, sleepQuality: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
              >
                <option value="Good">Good</option>
                <option value="Average">Average</option>
                <option value="Bad">Bad</option>
                <option value="Worst">Worst</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Study/Work Pressure</label>
              <select
                value={state.studyPressure}
                onChange={(e) => setState({ ...state, studyPressure: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
              >
                <option value="Good">Low</option>
                <option value="Average">Average</option>
                <option value="Bad">High</option>
                <option value="Worst">Very High</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Financial Pressure</label>
              <select
                value={state.financialPressure}
                onChange={(e) => setState({ ...state, financialPressure: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
              >
                <option value="Good">Low</option>
                <option value="Average">Average</option>
                <option value="Bad">High</option>
                <option value="Worst">Very High</option>
              </select>
            </div>
          </div>
        </div>

        {/* Questionnaire Form */}
        <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
          <div className="space-y-8">
            {PHQ9_QUESTIONS.map((question, index) => (
              <div key={index} className="border-b border-gray-100 pb-6 last:border-0">
                <label className="block text-base font-medium text-gray-900 mb-4">
                  {index + 1}. {question}
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {OPTIONS.map((option) => (
                    <label
                      key={option.value}
                      className={`flex items-center gap-3 p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                        state.answers[index] === option.value
                          ? 'border-primary-500 bg-primary-50'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      <input
                        type="radio"
                        name={`question-${index}`}
                        value={option.value}
                        checked={state.answers[index] === option.value}
                        onChange={() => handleAnswerChange(index, option.value)}
                        className="w-4 h-4 text-primary-500 focus:ring-primary-500 focus:ring-2"
                      />
                      <span className="text-sm font-medium text-gray-700">{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Total Score Display */}
          <div className="mt-8 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Total Score (out of 27):</span>
              <span className="text-2xl font-bold text-gray-900">{totalScore}</span>
            </div>
          </div>

          {/* Submit Button */}
          <div className="mt-8 flex items-center justify-between gap-4">
            <button
              type="button"
              onClick={() => navigate('/dashboard')}
              className="px-6 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              disabled={state.isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!isComplete() || state.isSubmitting}
              className={`px-6 py-3 rounded-lg font-medium transition-colors flex items-center gap-2 ${
                isComplete() && !state.isSubmitting
                  ? 'bg-primary-500 text-white hover:bg-primary-600'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              {state.isSubmitting ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  <CheckCircle2 className="w-5 h-5" />
                  Submit Assessment
                </>
              )}
            </button>
          </div>
        </form>

        {/* Information Footer */}
        <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-800">
            <strong>Note:</strong> This assessment is for informational purposes only and does not replace
            professional medical advice. If you're experiencing a mental health crisis, please contact a
            healthcare provider or emergency services immediately.
          </p>
        </div>
      </div>
    </div>
  );
};
