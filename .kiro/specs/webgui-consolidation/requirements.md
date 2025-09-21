# Requirements Document

## Introduction

This specification outlines the migration of Telegram bot management functionality from the existing Flask-based admin panel (`src/frontend/telegram/screener/admin_panel.py`) into the unified web UI system (`src/web_ui`). 

**Current State:**
- **Flask Admin Panel**: A comprehensive but outdated Flask-based web interface for managing the Telegram Screener Bot system
- **Target web_ui**: A modern Vite + React application for trading system management

**Flask Admin Panel Features to Migrate:**
- Dashboard with real-time statistics and user metrics
- User management (verification, approval, email reset, admin privileges)
- Alert management (create, toggle, delete price alerts with re-arm functionality)
- Schedule management (daily/weekly reports configuration)
- Feedback and feature request handling
- Broadcast messaging to all Telegram bot users
- Command audit logging and user activity tracking
- JSON configuration generator for alerts, schedules, and screeners

The goal is to replace the Flask admin panel with modern React components integrated into the existing `src/web_ui` system, providing a unified interface for both trading operations and Telegram bot management.

## Requirements

### Requirement 1

**User Story:** As a trading system administrator, I want a unified web interface that consolidates all management functions, so that I can manage Telegram bots, trading operations, and system monitoring from a single modern application.

#### Acceptance Criteria

1. WHEN I access the web UI THEN I SHALL see a unified navigation that includes Telegram bot management, trading system management, and monitoring features
2. WHEN I navigate between different sections THEN the system SHALL maintain consistent authentication and session state
3. WHEN I perform operations THEN the system SHALL use a unified backend API architecture that communicates with both trading and Telegram bot services
4. IF I have appropriate permissions THEN I SHALL be able to access all consolidated features from the same interface

### Requirement 2

**User Story:** As a Telegram bot administrator, I want a modern React-based dashboard to replace the Flask admin panel, so that I can manage all Telegram bot operations with improved user experience and performance.

#### Acceptance Criteria

1. WHEN I access the Telegram bot dashboard THEN I SHALL see real-time statistics including total users, verified users, approved users, pending approvals, active alerts, active schedules, and command usage metrics
2. WHEN I view user management THEN I SHALL be able to list, filter, verify, approve, reject, and reset email for Telegram bot users
3. WHEN I manage alerts THEN I SHALL be able to view, create, toggle, delete, and configure re-arm settings for price alerts
4. WHEN I manage schedules THEN I SHALL be able to view, create, modify, and delete daily/weekly report schedules
5. WHEN I need to broadcast messages THEN I SHALL be able to send announcements to all registered Telegram bot users
6. WHEN I review user activity THEN I SHALL be able to view command audit logs with filtering by user, time period, and command type

### Requirement 3

**User Story:** As a developer maintaining the system, I want the Telegram bot management interface to use the same modern technology stack as the trading system, so that the codebase is consistent and maintainable.

#### Acceptance Criteria

1. WHEN implementing Telegram bot management THEN the system SHALL use the existing Vite + React architecture from `src/web_ui`
2. WHEN creating UI components THEN the system SHALL use Material-UI v5 for consistent styling with the trading interface
3. WHEN handling server state THEN the system SHALL use React Query (@tanstack/react-query) for API calls and caching
4. WHEN managing client state THEN the system SHALL use Zustand for local state management
5. WHEN implementing real-time features THEN the system SHALL use Socket.io for WebSocket communication with the Telegram bot backend
6. WHEN authentication is required THEN the system SHALL integrate with the existing FastAPI backend authentication system

### Requirement 4

**User Story:** As a system administrator, I want seamless navigation between Telegram bot management and trading operations within the unified interface, so that I can efficiently manage all system aspects without switching applications.

#### Acceptance Criteria

1. WHEN I navigate between Telegram bot and trading sections THEN the system SHALL provide smooth client-side routing transitions using React Router
2. WHEN I access the main navigation THEN I SHALL see clearly organized sections for Trading Operations and Telegram Bot Management
3. WHEN I am in a specific section THEN the system SHALL show contextual breadcrumbs and highlight the current location in the navigation
4. WHEN I bookmark specific pages THEN the system SHALL support deep linking to all Telegram bot management pages (dashboard, users, alerts, schedules, etc.)
5. IF I need to cross-reference data THEN the system SHALL provide logical navigation paths between related Telegram bot and trading features

### Requirement 5

**User Story:** As a system operator, I want real-time updates for Telegram bot activities and user interactions, so that I can monitor bot performance and user engagement in real-time.

#### Acceptance Criteria

1. WHEN Telegram bot commands are executed THEN I SHALL see real-time command activity updates in the dashboard
2. WHEN users register, verify, or request approval THEN I SHALL receive live notifications in the user management interface
3. WHEN alerts are triggered or schedules execute THEN I SHALL see real-time status updates in the respective management sections
4. WHEN broadcast messages are sent THEN I SHALL see live delivery status and success/failure counts
5. WHEN viewing the dashboard THEN statistics SHALL update automatically without requiring page refresh
6. IF I have multiple browser tabs open THEN real-time updates SHALL be synchronized across all tabs using WebSocket connections

### Requirement 6

**User Story:** As a system administrator, I want comprehensive user management capabilities for Telegram bot users, so that I can control access, monitor activity, and maintain system security.

#### Acceptance Criteria

1. WHEN I view user lists THEN I SHALL be able to filter by verification status, approval status, and admin privileges
2. WHEN I manage user permissions THEN I SHALL be able to verify emails, approve/reject access requests, and grant admin privileges
3. WHEN I need to troubleshoot user issues THEN I SHALL be able to reset user emails and view individual user command history
4. WHEN I audit system usage THEN I SHALL see comprehensive command logs with user identification, timestamps, and command details
5. WHEN I set system limits THEN I SHALL be able to configure maximum alerts and schedules per user globally or individually
6. IF I need to investigate suspicious activity THEN I SHALL be able to view commands from non-registered users and take appropriate action

### Requirement 7

**User Story:** As a system administrator, I want comprehensive alert and schedule management capabilities, so that I can configure and monitor all automated Telegram bot features.

#### Acceptance Criteria

1. WHEN I manage alerts THEN I SHALL be able to view all active and inactive price alerts with their current status and re-arm configurations
2. WHEN I create or modify alerts THEN I SHALL be able to configure alert parameters, re-arm settings, hysteresis values, and technical indicators
3. WHEN I manage schedules THEN I SHALL be able to view, create, modify, and delete daily/weekly report schedules for users
4. WHEN I monitor alert performance THEN I SHALL see alert trigger history, success rates, and re-arm cycle information
5. WHEN I need to troubleshoot THEN I SHALL be able to manually toggle alert states and view detailed configuration JSON
6. IF alerts malfunction THEN I SHALL be able to disable problematic alerts and investigate their configuration

### Requirement 8

**User Story:** As a system administrator, I want broadcast messaging capabilities, so that I can communicate important announcements to all Telegram bot users efficiently.

#### Acceptance Criteria

1. WHEN I need to send announcements THEN I SHALL be able to compose and send broadcast messages to all registered users
2. WHEN I send broadcasts THEN I SHALL see real-time delivery status showing success and failure counts
3. WHEN I review communication history THEN I SHALL be able to view previous broadcast messages with timestamps and delivery statistics
4. WHEN broadcasts are sent THEN the system SHALL log the activity for audit purposes with sender identification
5. IF broadcast delivery fails THEN I SHALL see detailed error information and retry options
6. WHEN composing messages THEN I SHALL have a user-friendly interface with message preview and character count

### Requirement 9

**User Story:** As a system administrator, I want improved performance and user experience compared to the current Flask admin panel, so that I can manage Telegram bot operations more efficiently.

#### Acceptance Criteria

1. WHEN I load the Telegram bot management interface THEN the initial load time SHALL be significantly faster than the current Flask application
2. WHEN I navigate between different management sections THEN transitions SHALL be smooth with client-side routing
3. WHEN I interact with forms and data tables THEN the system SHALL provide immediate feedback and validation
4. WHEN I work with large user lists or audit logs THEN the system SHALL implement efficient pagination and search functionality
5. WHEN I use the interface on mobile devices THEN all Telegram bot management features SHALL be responsive and touch-friendly
6. IF I need to perform bulk operations THEN the system SHALL provide batch actions for user management and alert configuration

### Requirement 10

**User Story:** As a system integrator, I want the new React interface to maintain compatibility with existing Telegram bot backend services, so that no backend modifications are required during the migration.

#### Acceptance Criteria

1. WHEN the React interface communicates with the Telegram bot backend THEN it SHALL use existing database service functions from `src.data.db.telegram_service`
2. WHEN real-time updates are needed THEN the system SHALL integrate with the existing bot API endpoints (e.g., `/api/broadcast`)
3. WHEN authentication is required THEN the system SHALL integrate with the existing FastAPI authentication system used by the trading interface
4. WHEN accessing Telegram bot data THEN the system SHALL use the same database connections and models as the current Flask admin panel
5. IF new API endpoints are needed THEN they SHALL follow the established FastAPI patterns used in the trading system backend
6. WHEN the migration is complete THEN the Flask admin panel SHALL be safely removable without affecting Telegram bot functionality