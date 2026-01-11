import { useState } from 'react';
import { Layout } from './Layout';
import { HRDashboard } from './dashboards/HRDashboard';
import { ManagerDashboard } from './dashboards/ManagerDashboard';
import { EmployeeDashboard } from './dashboards/EmployeeDashboard';
import { Role } from '../types';

export const DashboardPage = () => {
  const [currentRole, setCurrentRole] = useState<Role>('hr');

  const renderDashboard = () => {
    switch (currentRole) {
      case 'hr':
        return <HRDashboard />;
      case 'manager':
        return <ManagerDashboard />;
      case 'employee':
        return <EmployeeDashboard />;
      default:
        return <HRDashboard />;
    }
  };

  return (
    <Layout currentRole={currentRole} onRoleChange={setCurrentRole}>
      {renderDashboard()}
    </Layout>
  );
};
