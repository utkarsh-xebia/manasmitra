import { useState } from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { Role } from '../types';
import { SearchProvider } from '../context/SearchContext';

interface LayoutProps {
  children: React.ReactNode;
  currentRole: Role;
}

export const Layout: React.FC<LayoutProps> = ({ children, currentRole }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <SearchProvider>
      <div className="min-h-screen bg-gray-50">
        <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} currentRole={currentRole} />
        <div className="lg:pl-64">
          <Header
            currentRole={currentRole}
            onMenuClick={() => setSidebarOpen(true)}
          />
          <main className="p-6">{children}</main>
        </div>
      </div>
    </SearchProvider>
  );
};
