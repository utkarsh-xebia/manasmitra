import { useState, useEffect } from 'react';
import { Layout } from './Layout';
import { HRDashboard } from './dashboards/HRDashboard';
import { ManagerDashboard } from './dashboards/ManagerDashboard';
import { EmployeeDashboard } from './dashboards/EmployeeDashboard';
import { Role } from '../types';

export const DashboardPage = () => {
  // Get role from localStorage, defaulting to 'employee' if not set
  const [currentRole] = useState<Role>(() => {
    return (localStorage.getItem('userRole') as Role) || 'employee';
  });

  const renderDashboard = () => {
    switch (currentRole) {
      case 'hr':
        return <HRDashboard />;
      case 'manager':
        return <ManagerDashboard />;
      case 'employee':
        return <EmployeeDashboard />;
      default:
        return <EmployeeDashboard />;
    }
  };

  return (
    <Layout currentRole={currentRole}>
      {renderDashboard()}
    </Layout>
  );
};
