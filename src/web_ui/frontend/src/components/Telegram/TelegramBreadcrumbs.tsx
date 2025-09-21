/**
 * Telegram Breadcrumbs Component
 * 
 * Provides breadcrumb navigation for Telegram bot management pages
 */

import React from 'react';
import { Breadcrumbs, Link, Typography } from '@mui/material';
import { useLocation, useNavigate } from 'react-router-dom';
import { Home, Telegram } from '@mui/icons-material';

interface BreadcrumbItem {
  label: string;
  path?: string;
  icon?: React.ReactNode;
}

const TelegramBreadcrumbs: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const getBreadcrumbs = (): BreadcrumbItem[] => {
    const pathSegments = location.pathname.split('/').filter(Boolean);
    const breadcrumbs: BreadcrumbItem[] = [
      { label: 'Home', path: '/dashboard', icon: <Home sx={{ mr: 0.5 }} fontSize="inherit" /> }
    ];

    if (pathSegments[0] === 'telegram') {
      breadcrumbs.push({
        label: 'Telegram Bot',
        path: '/telegram/dashboard',
        icon: <Telegram sx={{ mr: 0.5 }} fontSize="inherit" />
      });

      if (pathSegments[1]) {
        const pageLabels: Record<string, string> = {
          dashboard: 'Dashboard',
          users: 'User Management',
          alerts: 'Alert Management',
          schedules: 'Schedule Management',
          broadcast: 'Broadcast Center',
          audit: 'Audit Logs'
        };

        const pageLabel = pageLabels[pathSegments[1]];
        if (pageLabel) {
          breadcrumbs.push({ label: pageLabel });
        }
      }
    }

    return breadcrumbs;
  };

  const breadcrumbs = getBreadcrumbs();

  return (
    <Breadcrumbs aria-label="breadcrumb" sx={{ mb: 2 }}>
      {breadcrumbs.map((item, index) => {
        const isLast = index === breadcrumbs.length - 1;
        
        if (isLast || !item.path) {
          return (
            <Typography
              key={item.label}
              color="text.primary"
              sx={{ display: 'flex', alignItems: 'center' }}
            >
              {item.icon}
              {item.label}
            </Typography>
          );
        }

        return (
          <Link
            key={item.label}
            color="inherit"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              if (item.path) {
                navigate(item.path);
              }
            }}
            sx={{ 
              display: 'flex', 
              alignItems: 'center',
              textDecoration: 'none',
              '&:hover': {
                textDecoration: 'underline'
              }
            }}
          >
            {item.icon}
            {item.label}
          </Link>
        );
      })}
    </Breadcrumbs>
  );
};

export default TelegramBreadcrumbs;