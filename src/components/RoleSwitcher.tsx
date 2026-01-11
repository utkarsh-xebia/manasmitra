import { Role } from '../types';

interface RoleSwitcherProps {
  currentRole: Role;
  onRoleChange: (role: Role) => void;
}

export const RoleSwitcher: React.FC<RoleSwitcherProps> = ({ currentRole, onRoleChange }) => {
  const roles: { value: Role; label: string }[] = [
    { value: 'hr', label: 'HR' },
    { value: 'manager', label: 'Manager' },
    { value: 'employee', label: 'Employee' },
  ];

  return (
    <div className="flex items-center gap-2 bg-white rounded-lg p-1 border border-gray-200">
      {roles.map((role) => (
        <button
          key={role.value}
          onClick={() => onRoleChange(role.value)}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
            currentRole === role.value
              ? 'bg-primary-500 text-white shadow-sm'
              : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          {role.label}
        </button>
      ))}
    </div>
  );
};
