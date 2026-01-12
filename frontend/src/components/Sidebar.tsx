import { LayoutDashboard, Users, BarChart3, Settings, ClipboardList, X } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { Role } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  currentRole: Role;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose, currentRole }) => {
  const location = useLocation();

  const getMenuItems = () => {
    const items = [
      { icon: LayoutDashboard, label: 'Dashboard', href: '/dashboard' },
    ];

    if (currentRole === 'hr') {
      items.push({ icon: Users, label: 'Departments', href: '/departments' });
    } else if (currentRole === 'manager') {
      items.push({ icon: Users, label: 'Team', href: '/team' });
    } else if (currentRole === 'employee') {
      items.push({ icon: ClipboardList, label: 'Questionnaire', href: '/questionnaire' });
      items.push({ icon: Users, label: 'Profile', href: '/profile' });
    }

    return items;
  };

  const menuItems = getMenuItems();

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-20 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 w-64 bg-white border-r border-gray-200 transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        <div className="flex flex-col h-full">
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">W</span>
              </div>
              <span className="text-xl font-bold text-gray-900">Wellbeing</span>
            </div>
            <button
              onClick={onClose}
              className="lg:hidden p-1 hover:bg-gray-100 rounded-lg"
              aria-label="Close menu"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>

          <nav className="flex-1 p-4 space-y-1">
            {menuItems.map((item, index) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={index}
                  to={item.href}
                  onClick={onClose}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-primary-50 text-primary-600 font-medium'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          <div className="p-4 border-t border-gray-200">
            <div className="px-4 py-3 bg-gray-50 rounded-lg">
              <p className="text-xs font-medium text-gray-500 uppercase mb-1">Current Role</p>
              <p className="text-sm font-semibold text-gray-900 capitalize">{currentRole}</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
