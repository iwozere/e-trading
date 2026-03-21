import React from 'react';
import { Box, Typography } from '@mui/material';
import TelegramBreadcrumbs from '../../components/Telegram/TelegramBreadcrumbs';
import ConfigBuilder from '../../components/Telegram/ConfigBuilder';

const ConfigBuilderPage: React.FC = () => {
  const handleSaveConfig = (config: any, mode: 'alert' | 'schedule') => {
    console.log('Saved config from builder page:', { mode, config });
    alert('Config generated successfully.');
  };

  return (
    <Box sx={{ p: 3 }}>
      <TelegramBreadcrumbs />
      <Box mb={3}>
        <Typography variant="h4" gutterBottom>
          Static Page Builder
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Visually construct complex alert and schedule configurations for the Telegram bot.
        </Typography>
      </Box>
      <ConfigBuilder 
        onSave={handleSaveConfig} 
        onCancel={() => {}} 
        initialMode="alert"
      />
    </Box>
  );
};

export default ConfigBuilderPage;
