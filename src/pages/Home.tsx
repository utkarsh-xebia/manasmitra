import { Link } from 'react-router-dom';

export const Home = () => {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
      <div className="max-w-2xl w-full text-center">
        <div className="mb-8">
          <div className="w-16 h-16 bg-primary-500 rounded-lg flex items-center justify-center mx-auto mb-4">
            <span className="text-white font-bold text-2xl">W</span>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Employee Wellbeing Dashboard</h1>
          <p className="text-xl text-gray-600">
            Track and improve employee wellbeing across your organization
          </p>
        </div>
        <div className="flex gap-4 justify-center">
          <Link
            to="/login"
            className="px-6 py-3 bg-primary-500 text-white rounded-lg font-medium hover:bg-primary-600 transition-colors"
          >
            Login
          </Link>
          <Link
            to="/signup"
            className="px-6 py-3 bg-white text-primary-500 border border-primary-500 rounded-lg font-medium hover:bg-primary-50 transition-colors"
          >
            Sign Up
          </Link>
        </div>
      </div>
    </div>
  );
};
