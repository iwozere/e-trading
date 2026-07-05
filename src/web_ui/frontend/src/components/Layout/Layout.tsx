import React from 'react';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Avatar,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard,
  TrendingUp,
  Monitor,
  Analytics,
  Settings,
  Email,
  AccountCircle,
  Logout,
  Telegram,
  People,
  NotificationsActive,
  Schedule,
  Campaign,
  History,
  ExpandLess,
  ExpandMore,
  LockOpen,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../../stores/authStore';
import { useTelegramPermissions } from '../Telegram/TelegramRouteGuard';
import { ChangePasswordModal } from './ChangePasswordModal';

const drawerWidth = 240;

// Injected at build time (see vite.config.ts). Fallbacks keep dev/tests happy.
const APP_VERSION = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : 'dev';
const BUILD_DATE =
  typeof __BUILD_TIME__ !== 'undefined' ? __BUILD_TIME__.slice(0, 10) : '';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  
  const [mobileOpen, setMobileOpen] = React.useState(false);
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [telegramExpanded, setTelegramExpanded] = React.useState(false);
  const [changePasswordOpen, setChangePasswordOpen] = React.useState(false);
  
  const telegramPermissions = useTelegramPermissions();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    await logout();
    handleClose();
  };

  const menuItems = [
    { text: 'Dashboard', icon: <Dashboard />, path: '/dashboard' },
    { text: 'Strategies', icon: <TrendingUp />, path: '/strategies' },
    { text: 'Monitoring', icon: <Monitor />, path: '/monitoring' },
    { text: 'Analytics', icon: <Analytics />, path: '/analytics' },
    { text: 'Messages', icon: <Email />, path: '/messages' },
    { text: 'Administration', icon: <Settings />, path: '/administration' },
  ];

  const telegramMenuItems = [
    { text: 'Dashboard', icon: <Dashboard />, path: '/telegram/dashboard', permission: 'view' },
    { text: 'Config Builder', icon: <Settings />, path: '/telegram/builder', permission: 'manage_alerts' },
    { text: 'User Management', icon: <People />, path: '/telegram/users', permission: 'manage_users' },
    { text: 'Alert Management', icon: <NotificationsActive />, path: '/telegram/alerts', permission: 'manage_alerts' },
    { text: 'Schedule Management', icon: <Schedule />, path: '/telegram/schedules', permission: 'manage_alerts' },
    { text: 'Broadcast Center', icon: <Campaign />, path: '/telegram/broadcast', permission: 'send_broadcasts' },
    { text: 'Audit Logs', icon: <History />, path: '/telegram/audit', permission: 'view_audit_logs' },
  ];

  const handleTelegramToggle = () => {
    setTelegramExpanded(!telegramExpanded);
  };

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          Trading System
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => navigate(item.path)}
              sx={{
                mb: 0.5,
                mx: 1,
                borderRadius: '8px',
                '&.Mui-selected': {
                  backgroundColor: 'rgba(0, 229, 255, 0.08)',
                  '&:before': {
                    content: '""',
                    position: 'absolute',
                    left: -8,
                    top: '20%',
                    bottom: '20%',
                    width: 4,
                    backgroundColor: 'primary.main',
                    borderRadius: '0 4px 4px 0',
                    boxShadow: '0 0 10px rgba(0, 229, 255, 0.5)',
                  }
                },
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.04)',
                }
              }}
            >
              <ListItemIcon sx={{ color: location.pathname === item.path ? 'primary.main' : 'inherit', minWidth: 40 }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text} 
                primaryTypographyProps={{ 
                  fontSize: '0.9rem', 
                  fontWeight: location.pathname === item.path ? 600 : 400 
                }} 
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      
      {/* Telegram Bot Management Section */}
      {telegramPermissions.canView && (
        <>
          <Divider />
          <List>
            <ListItem disablePadding>
              <ListItemButton onClick={handleTelegramToggle}>
                <ListItemIcon>
                  <Telegram />
                </ListItemIcon>
                <ListItemText primary="Telegram Bot" />
                {telegramExpanded ? <ExpandLess /> : <ExpandMore />}
              </ListItemButton>
            </ListItem>
            
            {telegramExpanded && telegramMenuItems.map((item) => {
              const hasPermission = telegramPermissions.hasPermission(item.permission);
              if (!hasPermission) return null;
              
              return (
                <ListItem key={item.text} disablePadding sx={{ pl: 2 }}>
                  <ListItemButton
                    selected={location.pathname === item.path}
                    onClick={() => navigate(item.path)}
                    sx={{
                      mb: 0.5,
                      mx: 1,
                      borderRadius: '8px',
                      '&.Mui-selected': {
                        backgroundColor: 'rgba(0, 229, 255, 0.08)',
                        '&:before': {
                          content: '""',
                          position: 'absolute',
                          left: -8,
                          top: '20%',
                          bottom: '20%',
                          width: 4,
                          backgroundColor: 'primary.main',
                          borderRadius: '0 4px 4px 0',
                          boxShadow: '0 0 10px rgba(0, 229, 255, 0.5)',
                        }
                      },
                      '&:hover': {
                        backgroundColor: 'rgba(255, 255, 255, 0.04)',
                      }
                    }}
                  >
                    <ListItemIcon sx={{ color: location.pathname === item.path ? 'primary.main' : 'inherit', minWidth: 40 }}>
                      {item.icon}
                    </ListItemIcon>
                    <ListItemText 
                      primary={item.text} 
                      primaryTypographyProps={{ 
                        fontSize: '0.85rem', 
                        fontWeight: location.pathname === item.path ? 600 : 400 
                      }} 
                    />
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
        </>
      )}

      {/* Version label (static, baked in at build time) */}
      <Box sx={{ mt: 'auto', p: 2 }}>
        <Divider sx={{ mb: 1 }} />
        <Typography variant="caption" color="text.secondary" noWrap display="block">
          v{APP_VERSION}
        </Typography>
        <Typography variant="caption" color="text.secondary" noWrap display="block">
          build {BUILD_DATE}
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography 
            variant="h6" 
            noWrap 
            component="div" 
            sx={{ 
              flexGrow: 1,
              fontFamily: 'Outfit, sans-serif',
              fontWeight: 600,
              background: 'linear-gradient(90deg, #fff 0%, rgba(255,255,255,0.7) 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Enhanced Multi-Strategy Trading System
          </Typography>
          <div>
            <IconButton
              size="large"
              aria-label="account of current user"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleMenu}
              color="inherit"
            >
              <Avatar sx={{ width: 32, height: 32 }}>
                {user?.username?.charAt(0).toUpperCase()}
              </Avatar>
            </IconButton>
            <Menu
              id="menu-appbar"
              anchorEl={anchorEl}
              anchorOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={Boolean(anchorEl)}
              onClose={handleClose}
            >
              <MenuItem onClick={handleClose}>
                <AccountCircle sx={{ mr: 1 }} />
                Profile
              </MenuItem>
              <MenuItem onClick={() => { handleClose(); setChangePasswordOpen(true); }}>
                <LockOpen sx={{ mr: 1 }} />
                Change Password
              </MenuItem>
              <MenuItem onClick={handleLogout}>
                <Logout sx={{ mr: 1 }} />
                Logout
              </MenuItem>
            </Menu>
          </div>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
        aria-label="mailbox folders"
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{ flexGrow: 1, width: { sm: `calc(100% - ${drawerWidth}px)` } }}
      >
        <Toolbar />
        {children}
      </Box>
      <ChangePasswordModal open={changePasswordOpen} onClose={() => setChangePasswordOpen(false)} />
    </Box>
  );
};

export default Layout;