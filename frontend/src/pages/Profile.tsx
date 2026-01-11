import { User as UserIcon, Mail, Shield, Calendar, MapPin } from 'lucide-react';

export const Profile = () => {
  const userDataString = localStorage.getItem('userData');
  const userData = userDataString ? JSON.parse(userDataString) : null;

  if (!userData) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <p className="text-gray-500 italic">No user data found. Please log in again.</p>
      </div>
    );
  }

  const profileFields = [
    { icon: UserIcon, label: 'Full Name', value: userData.name },
    { icon: Mail, label: 'Email Address', value: userData.email },
    { icon: Shield, label: 'Role', value: userData.role.toUpperCase() },
    { icon: Calendar, label: 'Joined', value: new Date().toLocaleDateString() }, // Mock date
    { icon: MapPin, label: 'Location', value: 'Global' }, // Mock location
  ];

  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        {/* Cover Section */}
        <div className="h-32 bg-gradient-to-r from-primary-500 to-primary-600"></div>
        
        {/* Profile Info */}
        <div className="relative px-8 pb-8">
          <div className="absolute -top-12 left-8">
            <div className="w-24 h-24 bg-white rounded-2xl p-1 shadow-lg">
              <div className="w-full h-full bg-primary-100 rounded-xl flex items-center justify-center">
                <UserIcon className="w-12 h-12 text-primary-600" />
              </div>
            </div>
          </div>
          
          <div className="pt-16">
            <h1 className="text-2xl font-bold text-gray-900">{userData.name}</h1>
            <p className="text-gray-500 capitalize">{userData.role}</p>
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
            {profileFields.map((field, index) => (
              <div key={index} className="flex items-center gap-4 p-4 rounded-lg bg-gray-50 border border-gray-100">
                <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center shadow-sm">
                  <field.icon className="w-5 h-5 text-gray-500" />
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase">{field.label}</p>
                  <p className="text-sm font-semibold text-gray-900">{field.value}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 pt-8 border-t border-gray-100">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Account Settings</h2>
            <div className="flex flex-wrap gap-4">
              <button className="px-6 py-2 bg-primary-500 text-white rounded-lg font-medium hover:bg-primary-600 transition-colors">
                Edit Profile
              </button>
              <button className="px-6 py-2 bg-white text-gray-700 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 transition-colors">
                Change Password
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
