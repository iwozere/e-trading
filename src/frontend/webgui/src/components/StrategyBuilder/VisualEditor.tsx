import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

const VisualEditor: React.FC = () => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6">Strategy Visual Editor</Typography>
        <Box sx={{ height: 220, background: '#f0f4f8', border: '2px dashed #90caf9', borderRadius: 2, mt: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography color="text.secondary">[Drag & Drop strategy builder coming soon]</Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default VisualEditor; 