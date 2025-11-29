/**
 * StatCard Component
 * 
 * Reusable card component for displaying key metrics with customizable
 * colors, actions, and visual feedback.
 */

import React from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  Skeleton,
  IconButton,
  Tooltip,
  Fade,
  useTheme
} from '@mui/material';
import { SxProps, Theme } from '@mui/material/styles';
import { ArrowForward as ArrowForwardIcon } from '@mui/icons-material';

export interface StatCardProps {
  /**
   * Title of the statistic
   */
  title: string;
  
  /**
   * Main value to display
   */
  value: number | string | undefined;
  
  /**
   * Optional subtitle or additional context
   */
  subtitle?: string;
  
  /**
   * Color theme for the card
   */
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  
  /**
   * Optional icon to display
   */
  icon?: React.ReactNode;
  
  /**
   * Optional action button or custom action element
   */
  action?: React.ReactNode;
  
  /**
   * Loading state
   */
  loading?: boolean;
  
  /**
   * Click handler for navigation
   */
  onClick?: () => void;
  
  /**
   * Additional styles
   */
  sx?: SxProps<Theme>;
  
  /**
   * Tooltip text for the card
   */
  tooltip?: string;
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  color = 'primary',
  icon,
  action,
  loading = false,
  onClick,
  sx,
  tooltip
}) => {
  const theme = useTheme();
  const isClickable = !!onClick;

  const getColorValue = (colorName: string) => {
    switch (colorName) {
      case 'primary':
        return theme.palette.primary.main;
      case 'secondary':
        return theme.palette.secondary.main;
      case 'success':
        return theme.palette.success.main;
      case 'warning':
        return theme.palette.warning.main;
      case 'error':
        return theme.palette.error.main;
      case 'info':
        return theme.palette.info.main;
      default:
        return theme.palette.primary.main;
    }
  };

  const cardContent = (
    <Card
      sx={{
        height: '100%',
        cursor: isClickable ? 'pointer' : 'default',
        transition: 'all 0.3s ease',
        '&:hover': isClickable ? {
          transform: 'translateY(-4px)',
          boxShadow: theme.shadows[8],
        } : {},
        ...sx
      }}
      onClick={onClick}
    >
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box flex={1}>
            {/* Title and Icon */}
            <Box display="flex" alignItems="center" mb={1}>
              {icon && (
                <Box
                  sx={{
                    mr: 1,
                    color: getColorValue(color),
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  {icon}
                </Box>
              )}
              <Typography
                color="textSecondary"
                variant="body2"
                fontWeight={500}
              >
                {title}
              </Typography>
            </Box>

            {/* Value */}
            {loading ? (
              <Skeleton variant="text" width="60%" height={48} />
            ) : (
              <Fade in={!loading} timeout={500}>
                <Typography
                  variant="h4"
                  sx={{
                    color: getColorValue(color),
                    fontWeight: 600,
                    mb: subtitle ? 0.5 : 0
                  }}
                >
                  {value ?? '-'}
                </Typography>
              </Fade>
            )}

            {/* Subtitle */}
            {subtitle && (
              <Typography
                color="textSecondary"
                variant="body2"
                sx={{ mt: 0.5 }}
              >
                {subtitle}
              </Typography>
            )}
          </Box>

          {/* Action */}
          {action && (
            <Box>
              {action}
            </Box>
          )}

          {/* Default navigation icon for clickable cards */}
          {isClickable && !action && (
            <IconButton
              size="small"
              sx={{
                color: getColorValue(color),
                opacity: 0.7,
                '&:hover': {
                  opacity: 1,
                  backgroundColor: 'transparent'
                }
              }}
            >
              <ArrowForwardIcon />
            </IconButton>
          )}
        </Box>
      </CardContent>
    </Card>
  );

  if (tooltip) {
    return (
      <Tooltip title={tooltip} arrow placement="top">
        {cardContent}
      </Tooltip>
    );
  }

  return cardContent;
};

export default StatCard;
