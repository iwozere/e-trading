import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  Container,
  Avatar,
  Link,
  IconButton,
  InputAdornment,
} from '@mui/material';
import { LockOutlined, Visibility, VisibilityOff, Telegram, Email } from '@mui/icons-material';
import { useAuthStore } from '../../stores/authStore';
import toast from 'react-hot-toast';

type AuthMode = 'login' | 'forgot' | 'reset';

const Login: React.FC = () => {
  const [mode, setMode] = useState<AuthMode>('login');
  
  // Login fields
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  
  // Forgot/Reset fields
  const [identity, setIdentity] = useState('');
  const [code, setCode] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  
  const [showPassword, setShowPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  
  const { login } = useAuthStore();

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const success = await login(username, password);
      if (!success) {
        setError('Invalid username or password');
      }
    } catch (err) {
      setError('Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleForgotSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    try {
      const response = await fetch('/auth/reset-password/request', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ identity }),
      });

      const data = await response.json();

      if (response.ok) {
        const dest = data.channel === 'telegram' ? 'Telegram bot' : 'Email inbox';
        toast.success(`Reset code sent to your ${dest}`);
        setSuccess(`A verification code was sent to your ${data.recipient || 'account'}.`);
        setMode('reset');
      } else {
        setError(data.detail || 'Failed to send reset code');
      }
    } catch (err) {
      setError('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleResetSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (newPassword.length < 6) {
      setError('New password must be at least 6 characters long');
      setLoading(false);
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    try {
      const response = await fetch('/auth/reset-password/confirm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          identity,
          code,
          new_password: newPassword,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        toast.success('Password reset successfully. You can now log in.');
        setMode('login');
        setPassword(''); // Clear passwords
        setNewPassword('');
        setConfirmPassword('');
        setCode('');
      } else {
        setError(data.detail || 'Failed to reset password');
      }
    } catch (err) {
      setError('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container component="main" maxWidth="xs">
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Avatar sx={{ m: 1, bgcolor: 'primary.main' }}>
          <LockOutlined />
        </Avatar>
        
        <Typography component="h1" variant="h5" sx={{ fontFamily: 'Outfit, sans-serif', fontWeight: 600 }}>
          {mode === 'login' && 'Trading System Login'}
          {mode === 'forgot' && 'Reset Password'}
          {mode === 'reset' && 'Enter Verification Code'}
        </Typography>
        
        <Card sx={{ mt: 3, width: '100%', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
          <CardContent>
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            {success && (
              <Alert severity="success" sx={{ mb: 2 }}>
                {success}
              </Alert>
            )}

            {mode === 'login' && (
              <Box component="form" onSubmit={handleLoginSubmit} sx={{ mt: 1 }}>
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="username"
                  label="Username or Email"
                  name="username"
                  autoComplete="username"
                  autoFocus
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  disabled={loading}
                />
                
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  name="password"
                  label="Password"
                  type={showPassword ? 'text' : 'password'}
                  id="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={loading}
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          aria-label="toggle password visibility"
                          onClick={() => setShowPassword(!showPassword)}
                          edge="end"
                        >
                          {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />
                
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  sx={{ mt: 3, mb: 2 }}
                  disabled={loading}
                >
                  {loading ? 'Signing In...' : 'Sign In'}
                </Button>

                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                  <Link
                    component="button"
                    variant="body2"
                    type="button"
                    onClick={() => {
                      setMode('forgot');
                      setError('');
                      setSuccess('');
                    }}
                    sx={{ color: 'primary.light', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
                  >
                    Forgot password? Reset via Telegram / Email
                  </Link>
                </Box>
                
                <Box sx={{ mt: 3, p: 2, bgcolor: 'background.default', borderRadius: 1, border: '1px solid rgba(255, 255, 255, 0.04)' }}>
                  <Typography variant="body2" color="textSecondary" align="center">
                    <strong>Demo Credentials:</strong>
                  </Typography>
                  <Typography variant="body2" color="textSecondary" align="center">
                    Username: admin | Password: admin
                  </Typography>
                  <Typography variant="body2" color="textSecondary" align="center">
                    Username: trader | Password: trader
                  </Typography>
                </Box>
              </Box>
            )}

            {mode === 'forgot' && (
              <Box component="form" onSubmit={handleForgotSubmit} sx={{ mt: 1 }}>
                <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                  Enter your registered Email address or Telegram User ID. We will send a secure verification code to reset your password.
                </Typography>

                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="identity"
                  label="Email or Telegram ID"
                  name="identity"
                  autoFocus
                  placeholder="e.g. user@test.com or 12345678"
                  value={identity}
                  onChange={(e) => setIdentity(e.target.value)}
                  disabled={loading}
                />
                
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  sx={{ mt: 3, mb: 2 }}
                  disabled={loading}
                >
                  {loading ? 'Sending Code...' : 'Send Reset Code'}
                </Button>

                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                  <Link
                    component="button"
                    variant="body2"
                    type="button"
                    onClick={() => {
                      setMode('login');
                      setError('');
                      setSuccess('');
                    }}
                    sx={{ color: 'text.secondary', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
                  >
                    Back to Sign In
                  </Link>
                </Box>
              </Box>
            )}

            {mode === 'reset' && (
              <Box component="form" onSubmit={handleResetSubmit} sx={{ mt: 1 }}>
                <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                  A verification code has been dispatched. Enter it below along with your new password.
                </Typography>

                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="code"
                  label="Verification Code"
                  name="code"
                  autoFocus
                  placeholder="6-digit code"
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  disabled={loading}
                />

                <TextField
                  margin="normal"
                  required
                  fullWidth
                  name="newPassword"
                  label="New Password"
                  type={showNewPassword ? 'text' : 'password'}
                  id="newPassword"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  disabled={loading}
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          aria-label="toggle password visibility"
                          onClick={() => setShowNewPassword(!showNewPassword)}
                          edge="end"
                        >
                          {showNewPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />

                <TextField
                  margin="normal"
                  required
                  fullWidth
                  name="confirmPassword"
                  label="Confirm New Password"
                  type={showNewPassword ? 'text' : 'password'}
                  id="confirmPassword"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  disabled={loading}
                />
                
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  sx={{ mt: 3, mb: 2 }}
                  disabled={loading}
                >
                  {loading ? 'Resetting Password...' : 'Reset Password'}
                </Button>

                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, mt: 1 }}>
                  <Link
                    component="button"
                    variant="body2"
                    type="button"
                    onClick={() => {
                      setMode('forgot');
                      setError('');
                      setSuccess('');
                    }}
                    sx={{ color: 'primary.light', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
                  >
                    Resend verification code
                  </Link>
                  <Link
                    component="button"
                    variant="body2"
                    type="button"
                    onClick={() => {
                      setMode('login');
                      setError('');
                      setSuccess('');
                    }}
                    sx={{ color: 'text.secondary', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
                  >
                    Back to Sign In
                  </Link>
                </Box>
              </Box>
            )}
          </CardContent>
        </Card>
      </Box>
    </Container>
  );
};

export default Login;