import { useState } from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { Role } from '../types';

interface LayoutProps {
  children: React.ReactNode;
  currentRole: Role;
  onRoleChange: (role: Role) => void;
}

export const Layout: React.FC<LayoutProps> = ({ children, currentRole, onRoleChange }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} currentRole={currentRole} />
      <div className="lg:pl-64">
        <Header
          currentRole={currentRole}
          onRoleChange={onRoleChange}
          onMenuClick={() => setSidebarOpen(true)}
        />
        <main>{children}</main>
      </div>
    </div>
  );
};
