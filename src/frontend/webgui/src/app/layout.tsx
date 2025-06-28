import './globals.css';
import type { Metadata } from 'next';
import NavBar from '../components/NavBar';
import { getServerSession } from 'next-auth';
import { redirect } from 'next/navigation';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';

export const metadata: Metadata = {
  title: 'Trading System Manager',
  description: 'Professional trading system management web application',
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const session = await getServerSession();
  if (!session) {
    redirect('/login');
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }
  return (
    <html lang="en">
      <body>
        <NavBar />
        {children}
      </body>
    </html>
  );
} 