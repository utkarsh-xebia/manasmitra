import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface BarChartProps {
  data: { name: string; [key: string]: string | number }[];
  dataKeys: { key: string; color: string; name: string }[];
  height?: number;
}

export const BarChart: React.FC<BarChartProps> = ({ data, dataKeys, height = 300 }) => {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
          <p className="font-semibold text-gray-900 mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsBarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="name" stroke="#6b7280" />
        <YAxis stroke="#6b7280" />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        {dataKeys.map((dataKey) => (
          <Bar key={dataKey.key} dataKey={dataKey.key} fill={dataKey.color} name={dataKey.name} radius={[8, 8, 0, 0]} />
        ))}
      </RechartsBarChart>
    </ResponsiveContainer>
  );
};
