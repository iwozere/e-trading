"use client";

import React, { useState } from 'react';
import { getProviders, signIn } from 'next-auth/react';
import { Button, Container, Typography, Box, TextField, Divider, Stack } from '@mui/material';
import GoogleIcon from '@mui/icons-material/Google';
import GitHubIcon from '@mui/icons-material/GitHub';
import FacebookIcon from '@mui/icons-material/Facebook';

const providerIcons: Record<string, React.ReactNode> = {
  google: <GoogleIcon sx={{ mr: 1 }} />, 
  github: <GitHubIcon sx={{ mr: 1 }} />, 
  facebook: <FacebookIcon sx={{ mr: 1 }} />
};

export default function LoginPage() {
  const [providers, setProviders] = React.useState<any>({});
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  React.useEffect(() => {
    getProviders().then(setProviders);
  }, []);

  const handleCredentials = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    const res = await signIn('credentials', {
      username,
      password,
      redirect: false,
    });
    if (res?.error) setError('Invalid credentials');
    if (res?.ok) window.location.href = '/';
  };

  return (
    <Container maxWidth="xs" sx={{ mt: 8 }}>
      <Box sx={{ p: 4, boxShadow: 3, borderRadius: 2, bgcolor: 'background.paper' }}>
        <Typography variant="h5" align="center" gutterBottom>Sign in to Trading System</Typography>
        <Stack spacing={2}>
          {Object.values(providers).filter((p: any) => p.id !== 'credentials').map((provider: any) => (
            <Button
              key={provider.id}
              variant="outlined"
              fullWidth
              startIcon={providerIcons[provider.id] || null}
              onClick={() => signIn(provider.id)}
            >
              Sign in with {provider.name}
            </Button>
          ))}
        </Stack>
        <Divider sx={{ my: 3 }}>or</Divider>
        <form onSubmit={handleCredentials}>
          <TextField
            label="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            fullWidth
            margin="normal"
            required
          />
          <TextField
            label="Password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            fullWidth
            margin="normal"
            required
          />
          {error && <Typography color="error" variant="body2">{error}</Typography>}
          <Button type="submit" variant="contained" color="primary" fullWidth sx={{ mt: 2 }}>
            Sign in with Credentials
          </Button>
        </form>
      </Box>
    </Container>
  );
} 