#!/usr/bin/env python3
"""
Telegram Screener Bot - Admin Panel

A Flask-based web interface for managing the Telegram Screener Bot system.
Provides comprehensive administrative tools for user management, alerts, schedules, and feedback.

Features:
- Dashboard with real-time statistics
- User management (verification, email reset, admin privileges)
- Alert management (create, toggle, delete price alerts)
- Schedule management (daily/weekly reports)
- Feedback and feature request handling
- Broadcast messaging to all users

Usage:
1. Run the script: python admin_panel.py
2. Open your web browser and navigate to: http://localhost:5000
3. Login with admin credentials (configured in environment variables)
4. Use the navigation menu to access different admin functions

URL Structure:
- Login: http://localhost:5000/login
- Dashboard: http://localhost:5000/
- Users: http://localhost:5000/users
- Alerts: http://localhost:5000/alerts
- Schedules: http://localhost:5000/schedules
- Feedback: http://localhost:5000/feedback
- Broadcast: http://localhost:5000/broadcast
- Logout: http://localhost:5000/logout

Configuration:
- Default port: 5000
- Host: 0.0.0.0 (accessible from any IP)
- Debug mode: Enabled for development
- Secret key: 'alkotrader' (change in production)
- Admin credentials: Set WEBGUI_LOGIN and WEBGUI_PASSWORD environment variables

Security Note:
- Change the secret key in production
- Set strong admin credentials in environment variables
- Use HTTPS in production environments
- All admin routes are protected with login authentication

Dependencies:
- Flask
- SQLite3 (for database operations)
- Custom database module (src.frontend.telegram.db)
- Custom logger (src.notification.logger)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash, session
import sqlite3
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
from src.frontend.telegram import db
from config.donotshare.donotshare import WEBGUI_LOGIN, WEBGUI_PASSWORD, WEBGUI_PORT, TELEGRAM_BOT_TOKEN

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

app = Flask(__name__)

# Bot API configuration
BOT_API_URL = "http://localhost:8080"

# Authentication functions
def login_required(f):
    """Decorator to require login for protected routes"""
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def check_credentials(username, password):
    """Check if provided credentials match admin credentials from config"""
    if not WEBGUI_LOGIN or not WEBGUI_PASSWORD:
        _logger.error("Admin credentials not configured in environment variables")
        return False

    return username == WEBGUI_LOGIN and password == WEBGUI_PASSWORD

# TODO: Change this to a random secret key in production and put it into the .env file
app.secret_key = 'alkotrader'

async def send_broadcast_message(message_text: str, sent_by: str = "admin") -> tuple[int, int]:
    """Send broadcast message to all registered users using bot API. Returns (success_count, total_count)."""
    try:
        # Use the bot API to send broadcast
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{BOT_API_URL}/api/broadcast", json={
                "message": message_text,
                "title": "Alkotrader Announcement"
            }) as response:
                if response.status == 200:
                    result = await response.json()
                    success_count = result.get('success_count', 0)
                    total_count = result.get('total_count', 0)

                    _logger.info("Broadcast API call successful: %d/%d messages queued", success_count, total_count)

                    # Log broadcast to database for history
                    try:
                        conn = sqlite3.connect(db.DB_PATH)
                        c = conn.cursor()
                        c.execute("""INSERT INTO broadcast_log
                                    (message, sent_by, success_count, total_count, created)
                                    VALUES (?, ?, ?, ?, ?)""",
                                (message_text[:500], sent_by, success_count, total_count, datetime.now().isoformat()))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        _logger.error("Failed to log broadcast to database: %s", e)

                    return success_count, total_count
                else:
                    error_text = await response.text()
                    _logger.error("Broadcast API call failed: HTTP %d - %s", response.status, error_text)
                    return 0, 0

    except Exception as e:
        _logger.exception("Error in broadcast function: ")
        return 0, 0

# HTML Templates
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Telegram Screener Bot - Admin Login</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; height: 100vh; display: flex; align-items: center; justify-content: center; }
        .login-container { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); width: 100%; max-width: 400px; }
        .login-header { text-align: center; margin-bottom: 30px; }
        .login-header h1 { color: #333; margin: 0; font-size: 24px; }
        .login-header p { color: #666; margin: 10px 0 0 0; }
        .form-group { margin: 20px 0; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: bold; color: #333; }
        .form-group input { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 16px; box-sizing: border-box; }
        .form-group input:focus { outline: none; border-color: #007bff; box-shadow: 0 0 0 2px rgba(0,123,255,0.25); }
        .btn-login { width: 100%; background: #007bff; color: white; padding: 12px; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; margin-top: 10px; }
        .btn-login:hover { background: #0056b3; }
        .alert { padding: 12px; margin: 10px 0; border-radius: 4px; }
        .alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-header">
            <h1>Admin Panel</h1>
            <p>Telegram Screener Bot</p>
        </div>

        {% if error %}
        <div class="alert alert-error">{{ error }}</div>
        {% endif %}

        <form method="POST" action="{{ url_for('login') }}">
            <div class="form-group">
                <label for="username">Username:</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" class="btn-login">Login</button>
        </form>
    </div>
</body>
</html>
"""

ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Telegram Screener Bot - Admin Panel</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1600px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .nav { background: #333; color: white; padding: 15px; margin: -20px -20px 20px -20px; border-radius: 8px 8px 0 0; display: flex; justify-content: space-between; align-items: center; }
        .nav-left { display: flex; }
        .nav-right { display: flex; align-items: center; }
        .nav a { color: white; text-decoration: none; margin-right: 20px; }
        .nav a:hover { text-decoration: underline; }
        .nav .logout { color: #ff6b6b; }
        .nav .logout:hover { color: #ff5252; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        tr:hover { background-color: #f5f5f5; }
        .btn { background: #007bff; color: white; padding: 6px 12px; text-decoration: none; border-radius: 4px; border: none; cursor: pointer; font-size: 12px; margin: 2px; display: inline-block; }
        .btn:hover { background: #0056b3; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #218838; }
        .btn-secondary { background: #6c757d; color: white; }
        .btn-secondary:hover { background: #5a6268; }
        .btn-group { display: flex; flex-wrap: wrap; gap: 4px; }
        .btn-compact { padding: 4px 8px; font-size: 11px; }
        .user-table { font-size: 14px; }
        .user-table th, .user-table td { padding: 8px 6px; }
        .user-table .actions-cell { min-width: 200px; }
        .user-table .status-cell { text-align: center; }
        .user-table .id-cell { font-family: monospace; font-size: 12px; }
        .user-table .json-cell { font-family: monospace; font-size: 11px; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .user-table .json-cell:hover { white-space: normal; word-break: break-all; }
        .json-expandable { cursor: pointer; color: #007bff; text-decoration: underline; }
        .json-expandable:hover { color: #0056b3; }
        .json-full { display: none; background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 5px; font-family: monospace; font-size: 11px; white-space: pre-wrap; word-break: break-all; }
        .json-full.show { display: block; }
        .stats { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 8px; flex: 1; text-align: center; min-width: 200px; }
        .stat-number { font-size: 2em; font-weight: bold; color: #007bff; }
        .stat-links { margin-top: 10px; display: flex; gap: 5px; justify-content: center; flex-wrap: wrap; }
        .stat-link { font-size: 11px; color: #007bff; text-decoration: none; padding: 2px 6px; border: 1px solid #007bff; border-radius: 3px; background: #f8f9fa; transition: all 0.2s; }
        .stat-link:hover { background: #007bff; color: white; text-decoration: none; }
        .form-group { margin: 15px 0; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .alert { padding: 12px; margin: 10px 0; border-radius: 4px; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <div class="nav-left">
                <h2 style="margin: 0;">Telegram Screener Bot - Admin Panel</h2>
            </div>
            <div class="nav-right">
                <a href="{{ url_for('dashboard') }}">Dashboard</a>
                <a href="{{ url_for('users') }}">Users</a>
                <a href="{{ url_for('alerts') }}">Alerts</a>
                <a href="{{ url_for('schedules') }}">Schedules</a>
                <a href="{{ url_for('feedback') }}">Feedback</a>
                <a href="{{ url_for('audit') }}">Audit</a>
                <a href="{{ url_for('broadcast') }}">Broadcast</a>
                <a href="{{ url_for('help_page') }}" style="color: #28a745;">📖 Help</a>
                <span style="color: #ccc; margin: 0 15px;">Welcome, {{ session.get('username', 'Admin') }}</span>
                <a href="{{ url_for('logout') }}" class="logout">Logout</a>
            </div>
        </div>

        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="alert alert-success">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        {{ content | safe }}
    </div>

    <script>
        // JSON expansion functionality
        document.addEventListener('DOMContentLoaded', function() {
            // Add click handlers to JSON cells
            document.querySelectorAll('.json-cell').forEach(function(cell) {
                const jsonData = cell.getAttribute('title');
                if (jsonData && jsonData !== 'No JSON config') {
                    cell.classList.add('json-expandable');
                    cell.addEventListener('click', function() {
                        // Create or toggle full JSON display
                        let fullJson = cell.querySelector('.json-full');
                        if (!fullJson) {
                            fullJson = document.createElement('div');
                            fullJson.className = 'json-full';
                            try {
                                const parsed = JSON.parse(jsonData);
                                fullJson.textContent = JSON.stringify(parsed, null, 2);
                            } catch (e) {
                                fullJson.textContent = jsonData;
                            }
                            cell.appendChild(fullJson);
                        }
                        fullJson.classList.toggle('show');
                    });
                }
            });
        });
    </script>
</body>
</html>
"""

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle admin login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if check_credentials(username, password):
            session['logged_in'] = True
            session['username'] = username
            _logger.info("Admin login successful for user: %s", username)
            return redirect(url_for('dashboard'))
        else:
            _logger.warning("Failed login attempt for username: %s", username)
            return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password")

    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    """Handle admin logout"""
    session.clear()
    _logger.info("Admin logout successful")
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    """Admin dashboard with statistics."""
    try:
        db.init_db()

        # Get statistics
        users = db.list_users()
        total_users = len(users)
        verified_users = sum(1 for user in users if user['verified'])
        approved_users = sum(1 for user in users if user['approved'])
        pending_approvals = sum(1 for user in users if user['verified'] and not user['approved'])

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM alerts WHERE active=1")
        active_alerts = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM schedules WHERE active=1")
        active_schedules = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM feedback WHERE status='open'")
        open_feedback = c.fetchone()[0]

        # Get audit statistics
        audit_stats = db.get_command_audit_stats()

        conn.close()

        content = f"""
        <h3>Dashboard</h3>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{total_users}</div>
                <div>Total Users</div>
                <div class="stat-links">
                    <a href="/users" class="stat-link">View All</a>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{verified_users}</div>
                <div>Verified Users</div>
                <div class="stat-links">
                    <a href="/users?filter=verified" class="stat-link">View Verified</a>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{approved_users}</div>
                <div>Approved Users</div>
                <div class="stat-links">
                    <a href="/users?filter=approved" class="stat-link">View Approved</a>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{pending_approvals}</div>
                <div>Pending Approvals</div>
                <div class="stat-links">
                    <a href="/users?filter=pending" class="stat-link">View Pending</a>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{active_alerts}</div>
                <div>Active Alerts</div>
                <div class="stat-links">
                    <a href="/alerts?filter=active" class="stat-link">View Active</a>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{active_schedules}</div>
                <div>Active Schedules</div>
                <div class="stat-links">
                    <a href="/schedules?filter=active" class="stat-link">View Active</a>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{open_feedback}</div>
                <div>Open Feedback</div>
                <div class="stat-links">
                    <a href="/feedback?filter=open" class="stat-link">View Open</a>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{audit_stats['total_commands']}</div>
                <div>Total Commands</div>
                <div class="stat-links">
                    <a href="/audit" class="stat-link">View All</a>
                    <a href="/audit?period=24h" class="stat-link">Last 24h</a>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{audit_stats['recent_activity_24h']}</div>
                <div>Commands (24h)</div>
                <div class="stat-links">
                    <a href="/audit?period=24h" class="stat-link">View 24h</a>
                </div>
            </div>
        </div>

        <h4>Recent Activity</h4>
        <p>Welcome to the Telegram Screener Bot Admin Panel. Use the navigation above to manage users, alerts, schedules, and feedback.</p>

        {f'''
        <h4>Pending Approvals ({pending_approvals})</h4>
        <p>Users waiting for approval to access restricted features:</p>
        <table>
            <thead>
                <tr>
                    <th>Telegram ID</th>
                    <th>Email</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
        ''' + ''.join([f'''
                <tr>
                    <td>{user['telegram_user_id']}</td>
                    <td>{user['email'] or '(no email)'}</td>
                    <td>
                        <a href="/users/{user['telegram_user_id']}/approve" class="btn btn-success">Approve</a>
                        <a href="/users/{user['telegram_user_id']}/reject" class="btn btn-danger">Reject</a>
                    </td>
                </tr>
        ''' for user in users if user['verified'] and not user['approved']]) + '''
            </tbody>
        </table>
        ''' if pending_approvals > 0 else '<p>No pending approvals.</p>'}
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        _logger.exception("Error in dashboard: ")
        return f"Error: {str(e)}", 500

@app.route('/users')
@login_required
def users():
    """User management page."""
    try:
        db.init_db()
        users_list = db.list_users()

        # Apply filters
        filter_type = request.args.get('filter', '')
        if filter_type == 'verified':
            users_list = [u for u in users_list if u['verified']]
        elif filter_type == 'approved':
            users_list = [u for u in users_list if u['approved']]
        elif filter_type == 'pending':
            users_list = [u for u in users_list if u['verified'] and not u['approved']]

        filter_info = ""
        if filter_type:
            filter_info = f" (Filtered: {filter_type.title()})"

        content = f"""
        <h3>User Management{filter_info}</h3>

        <div style="margin-bottom: 20px;">
            <a href="/users" class="btn btn-secondary">All Users</a>
            <a href="/users?filter=verified" class="btn btn-secondary">Verified Only</a>
            <a href="/users?filter=approved" class="btn btn-secondary">Approved Only</a>
            <a href="/users?filter=pending" class="btn btn-secondary">Pending Approval</a>
        </div>

        <table class="user-table">
            <thead>
                <tr>
                    <th class="id-cell">Telegram ID</th>
                    <th>Email</th>
                    <th class="status-cell">Verified</th>
                    <th class="status-cell">Approved</th>
                    <th class="status-cell">Language</th>
                    <th class="status-cell">Admin</th>
                    <th class="status-cell">Max Alerts</th>
                    <th class="status-cell">Max Schedules</th>
                    <th class="actions-cell">Actions</th>
                </tr>
            </thead>
            <tbody>
        """

        for user in users_list:
            verified_badge = "✅" if user['verified'] else "❌"
            approved_badge = "✅" if user['approved'] else "❌"
            admin_badge = "👑" if user['is_admin'] else "👤"
            email = user['email'] or "(no email)"

            # Show Verify button only if user is not verified
            verify_button = f'<a href="/users/{user["telegram_user_id"]}/verify" class="btn btn-success btn-compact">Verify</a>' if not user['verified'] else '<span class="btn btn-secondary btn-compact" style="opacity: 0.5; cursor: not-allowed;">✓ Verified</span>'

            # Show Approve/Reject buttons only if user is verified but not approved
            if user['verified'] and not user['approved']:
                approve_button = f'<a href="/users/{user["telegram_user_id"]}/approve" class="btn btn-success btn-compact">Approve</a>'
                reject_button = f'<a href="/users/{user["telegram_user_id"]}/reject" class="btn btn-danger btn-compact">Reject</a>'
            elif user['approved']:
                approve_button = '<span class="btn btn-secondary btn-compact" style="opacity: 0.5; cursor: not-allowed;">✓ Approved</span>'
                reject_button = f'<a href="/users/{user["telegram_user_id"]}/reject" class="btn btn-danger btn-compact">Revoke</a>'
            else:
                approve_button = '<span class="btn btn-secondary btn-compact" style="opacity: 0.5; cursor: not-allowed;">Not Verified</span>'
                reject_button = '<span class="btn btn-secondary btn-compact" style="opacity: 0.5; cursor: not-allowed;">Not Verified</span>'

            # Show Reset Email button only if user has an email set
            reset_button = f'<a href="/users/{user["telegram_user_id"]}/reset" class="btn btn-danger btn-compact">Reset Email</a>' if user['email'] else '<span class="btn btn-secondary btn-compact" style="opacity: 0.5; cursor: not-allowed;">No Email</span>'

            # Add audit history button
            audit_button = f'<a href="/audit/user/{user["telegram_user_id"]}" class="btn btn-secondary btn-compact">History</a>'

            content += f"""
                <tr>
                    <td class="id-cell">{user['telegram_user_id']}</td>
                    <td>{email}</td>
                    <td class="status-cell">{verified_badge}</td>
                    <td class="status-cell">{approved_badge}</td>
                    <td class="status-cell">{user['language'] or 'en'}</td>
                    <td class="status-cell">{admin_badge}</td>
                    <td class="status-cell">{user['max_alerts']}</td>
                    <td class="status-cell">{user['max_schedules']}</td>
                    <td class="actions-cell">
                        <div class="btn-group">
                            {verify_button}
                            {approve_button}
                            {reject_button}
                            {reset_button}
                            {audit_button}
                        </div>
                    </td>
                </tr>
            """

        content += """
            </tbody>
        </table>
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        _logger.exception("Error in users page: ")
        return f"Error: {str(e)}", 500

@app.route('/users/<user_id>/verify')
@login_required
def verify_user(user_id):
    """Manually verify a user."""
    try:
        db.init_db()

        # Get current user status
        user = db.get_user_status(user_id)
        if not user:
            flash(f"User {user_id} not found.")
            return redirect(url_for('users'))

        if user['verified']:
            flash(f"User {user_id} is already verified.")
            return redirect(url_for('users'))

        db.update_user_verification(user_id, True)
        flash(f"User {user_id} has been manually verified.")
        return redirect(url_for('users'))
    except Exception as e:
        flash(f"Error verifying user: {str(e)}")
        return redirect(url_for('users'))

@app.route('/users/<user_id>/reset')
@login_required
def reset_user_email(user_id):
    """Reset a user's email."""
    try:
        db.init_db()

        # Get current user status
        user = db.get_user_status(user_id)
        if not user:
            flash(f"User {user_id} not found.")
            return redirect(url_for('users'))

        if not user['email']:
            flash(f"User {user_id} has no email to reset.")
            return redirect(url_for('users'))

        db.update_user_email(user_id, None)
        db.update_user_verification(user_id, False)
        flash(f"Email reset for user {user_id}.")
        return redirect(url_for('users'))
    except Exception as e:
        flash(f"Error resetting email: {str(e)}")
        return redirect(url_for('users'))

@app.route('/users/<user_id>/approve')
@login_required
def approve_user(user_id):
    """Approve a user for access to restricted features."""
    try:
        db.init_db()

        # Get current user status
        user = db.get_user_status(user_id)
        if not user:
            flash(f"User {user_id} not found.")
            return redirect(url_for('users'))

        if not user['verified']:
            flash(f"User {user_id} must be verified before approval.")
            return redirect(url_for('users'))

        if user['approved']:
            flash(f"User {user_id} is already approved.")
            return redirect(url_for('users'))

        db.approve_user(user_id)
        flash(f"User {user_id} ({user.get('email', 'no email')}) has been approved for restricted features.")
        return redirect(url_for('users'))
    except Exception as e:
        flash(f"Error approving user: {str(e)}")
        return redirect(url_for('users'))

@app.route('/users/<user_id>/reject')
@login_required
def reject_user(user_id):
    """Reject or revoke a user's approval."""
    try:
        db.init_db()

        # Get current user status
        user = db.get_user_status(user_id)
        if not user:
            flash(f"User {user_id} not found.")
            return redirect(url_for('users'))

        if not user['approved']:
            flash(f"User {user_id} is not approved.")
            return redirect(url_for('users'))

        db.reject_user(user_id)
        flash(f"User {user_id} ({user.get('email', 'no email')}) approval has been revoked.")
        return redirect(url_for('users'))
    except Exception as e:
        flash(f"Error rejecting user: {str(e)}")
        return redirect(url_for('users'))

@app.route('/alerts')
@login_required
def alerts():
    """Alert management page."""
    try:
        db.init_db()
        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM alerts ORDER BY created DESC")
        alerts_list = c.fetchall()
        conn.close()

        # Apply filters
        filter_type = request.args.get('filter', '')
        if filter_type == 'active':
            alerts_list = [a for a in alerts_list if a[5]]  # a[5] is the active field
        elif filter_type == 'advanced':
            alerts_list = [a for a in alerts_list if len(a) > 10 and a[10]]  # config_json field

        filter_info = ""
        if filter_type:
            filter_info = f" (Filtered: {filter_type.title()})"

        content = f"""
        <h3>Alert Management{filter_info}</h3>

        <div style="margin-bottom: 20px;">
            <a href="/alerts" class="btn btn-secondary">All Alerts</a>
            <a href="/alerts?filter=active" class="btn btn-secondary">Active Only</a>
            <a href="/alerts?filter=advanced" class="btn btn-secondary">JSON Alerts</a>
        </div>

        <table class="user-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>User ID</th>
                    <th>Ticker</th>
                    <th>Type</th>
                    <th>Price/Condition</th>
                    <th>Timeframe</th>
                    <th>Action</th>
                    <th>Email</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>JSON Config</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
        """

        for alert in alerts_list:
            status_badge = "🟢 Active" if alert[5] else "🔴 Inactive"
            email_badge = "📧" if alert[6] else "💬"

            # Get alert type and additional fields
            alert_type = alert[8] if len(alert) > 8 else "price"  # alert_type field
            timeframe = alert[9] if len(alert) > 9 else "15m"  # timeframe field
            config_json = alert[10] if len(alert) > 10 else None  # config_json field
            alert_action = alert[11] if len(alert) > 11 else "notify"  # alert_action field

            # Determine display values based on alert type
            if alert_type == "price":
                type_badge = "💰 Price"
                price_condition = f"${alert[3]:.2f} {alert[4]}"
            else:
                type_badge = "⚙️ Indicator"
                price_condition = "Complex"

            # Format JSON config for display
            json_display = ""
            if config_json:
                try:
                    import json
                    parsed_json = json.loads(config_json)
                    # Create a compact summary
                    if isinstance(parsed_json, dict):
                        summary_parts = []
                        if 'indicator' in parsed_json:
                            summary_parts.append(f"Indicator: {parsed_json['indicator']}")
                        if 'parameters' in parsed_json:
                            params = parsed_json['parameters']
                            if isinstance(params, dict):
                                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                                summary_parts.append(f"Params: {param_str}")
                        if 'condition' in parsed_json:
                            cond = parsed_json['condition']
                            if isinstance(cond, dict):
                                if 'operator' in cond and 'value' in cond:
                                    summary_parts.append(f"Condition: {cond['operator']} {cond['value']}")
                        json_display = " | ".join(summary_parts)
                    else:
                        json_display = str(parsed_json)[:50] + "..." if len(str(parsed_json)) > 50 else str(parsed_json)
                except Exception as e:
                    json_display = f"Error parsing JSON: {str(e)[:30]}..."

            content += f"""
                <tr>
                    <td>#{alert[0]}</td>
                    <td class="id-cell">{alert[2]}</td>
                    <td>{alert[1]}</td>
                    <td>{type_badge}</td>
                    <td>{price_condition}</td>
                    <td>{timeframe}</td>
                    <td>{alert_action}</td>
                    <td class="status-cell">{email_badge}</td>
                    <td class="status-cell">{status_badge}</td>
                    <td>{alert[7]}</td>
                    <td class="json-cell" title="{config_json or 'No JSON config'}">{json_display or 'N/A'}{' 🔍' if config_json else ''}</td>
                    <td>
                        <a href="/alerts/{alert[0]}/toggle" class="btn">Toggle</a>
                        <a href="/alerts/{alert[0]}/delete" class="btn btn-danger">Delete</a>
                    </td>
                </tr>
            """

        content += """
            </tbody>
        </table>
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        _logger.exception("Error in alerts page: ")
        return f"Error: {str(e)}", 500

@app.route('/schedules')
@login_required
def schedules():
    """Schedule management page."""
    try:
        db.init_db()
        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM schedules ORDER BY created DESC")
        schedules_list = c.fetchall()
        conn.close()

        # Apply filters
        filter_type = request.args.get('filter', '')
        if filter_type == 'active':
            schedules_list = [s for s in schedules_list if s[5]]  # s[5] is the active field
        elif filter_type == 'advanced':
            schedules_list = [s for s in schedules_list if len(s) > 13 and s[13]]  # config_json field

        filter_info = ""
        if filter_type:
            filter_info = f" (Filtered: {filter_type.title()})"

        content = f"""
        <h3>Schedule Management{filter_info}</h3>

        <div style="margin-bottom: 20px;">
            <a href="/schedules" class="btn btn-secondary">All Schedules</a>
            <a href="/schedules?filter=active" class="btn btn-secondary">Active Only</a>
            <a href="/schedules?filter=advanced" class="btn btn-secondary">JSON Schedules</a>
        </div>

        <table class="user-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>User ID</th>
                    <th>Type</th>
                    <th>Details</th>
                    <th>Time</th>
                    <th>Period</th>
                    <th>Interval</th>
                    <th>Provider</th>
                    <th>Email</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>JSON Config</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
        """

        for schedule in schedules_list:
            status_badge = "🟢 Active" if schedule[5] else "🔴 Inactive"
            email_badge = "📧" if schedule[6] else "💬"

            # Handle different schedule types
            schedule_config = schedule[14] if len(schedule) > 14 else "simple"  # schedule_config field
            config_json = schedule[13] if len(schedule) > 13 else None  # config_json field
            schedule_type = schedule[11] if len(schedule) > 11 else "report"  # schedule_type field
            list_type = schedule[12] if len(schedule) > 12 else None  # list_type field

            if schedule_config == "simple":
                # Simple schedule
                schedule_type_display = "📊 Report"
                details = f"{schedule[1] or 'N/A'}"
                time = schedule[3] or 'N/A'
            else:
                # JSON-based schedule
                try:
                    from src.frontend.telegram.screener.schedule_config_parser import get_schedule_summary
                    if config_json:
                        summary = get_schedule_summary(config_json)
                        if "error" not in summary:
                            schedule_type_display = f"⚙️ {summary.get('type', 'Unknown').title()}"
                            if summary.get('type') == 'report':
                                details = f"{summary.get('ticker', 'N/A')}"
                            elif summary.get('type') == 'screener':
                                details = f"{summary.get('list_type', 'N/A')}"
                            else:
                                details = "Custom"
                            time = summary.get('scheduled_time', 'N/A')
                        else:
                            schedule_type_display = "⚙️ JSON"
                            details = "Error parsing"
                            time = schedule[3] or 'N/A'
                    else:
                        schedule_type_display = "⚙️ JSON"
                        details = "No config"
                        time = schedule[3] or 'N/A'
                except Exception as e:
                    schedule_type_display = "⚙️ JSON"
                    details = "Parse error"
                    time = schedule[3] or 'N/A'

            # Format JSON config for display
            json_display = ""
            if config_json:
                try:
                    import json
                    parsed_json = json.loads(config_json)
                    # Create a compact summary
                    if isinstance(parsed_json, dict):
                        summary_parts = []
                        if 'type' in parsed_json:
                            summary_parts.append(f"Type: {parsed_json['type']}")
                        if 'ticker' in parsed_json:
                            summary_parts.append(f"Ticker: {parsed_json['ticker']}")
                        if 'list_type' in parsed_json:
                            summary_parts.append(f"List: {parsed_json['list_type']}")
                        if 'scheduled_time' in parsed_json:
                            summary_parts.append(f"Time: {parsed_json['scheduled_time']}")
                        if 'period' in parsed_json:
                            summary_parts.append(f"Period: {parsed_json['period']}")
                        if 'interval' in parsed_json:
                            summary_parts.append(f"Interval: {parsed_json['interval']}")
                        if 'indicators' in parsed_json:
                            indicators = parsed_json['indicators']
                            if isinstance(indicators, list):
                                summary_parts.append(f"Indicators: {', '.join(indicators)}")
                            elif isinstance(indicators, str):
                                summary_parts.append(f"Indicators: {indicators}")
                        json_display = " | ".join(summary_parts)
                    else:
                        json_display = str(parsed_json)[:50] + "..." if len(str(parsed_json)) > 50 else str(parsed_json)
                except Exception as e:
                    json_display = f"Error parsing JSON: {str(e)[:30]}..."

            content += f"""
                <tr>
                    <td>#{schedule[0]}</td>
                    <td class="id-cell">{schedule[2]}</td>
                    <td>{schedule_type_display}</td>
                    <td>{details}</td>
                    <td>{time}</td>
                    <td>{schedule[4] or 'N/A'}</td>
                    <td>{schedule[8] or 'N/A'}</td>
                    <td>{schedule[9] or 'N/A'}</td>
                    <td class="status-cell">{email_badge}</td>
                    <td class="status-cell">{status_badge}</td>
                    <td>{schedule[10]}</td>
                    <td class="json-cell" title="{config_json or 'No JSON config'}">{json_display or 'N/A'}{' 🔍' if config_json else ''}</td>
                    <td>
                        <a href="/schedules/{schedule[0]}/toggle" class="btn">Toggle</a>
                        <a href="/schedules/{schedule[0]}/delete" class="btn btn-danger">Delete</a>
                    </td>
                </tr>
            """

        content += """
            </tbody>
        </table>
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        _logger.exception("Error in schedules page: ")
        return f"Error: {str(e)}", 500

@app.route('/help')
def help_page():
    """Comprehensive help page accessible to all users."""
    try:
        content = """
        <h2>🤖 Alkotrader Bot - Complete Command Guide</h2>

        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h3>📋 Quick Start</h3>
            <p>Welcome to Alkotrader Bot! This bot provides comprehensive trading analysis, alerts, and scheduled reports.</p>
            <p><strong>Getting Started:</strong></p>
            <ol>
                <li>Send <code>/start</code> to begin</li>
                <li>Register with <code>/register your@email.com</code></li>
                <li>Verify your email with the code sent to you</li>
                <li>Request approval with <code>/request_approval</code></li>
                <li>Start using the bot's features!</li>
            </ol>
        </div>

        <h3>📊 Report Commands</h3>
        <div style="background: #e8f5e8; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h4>/report TICKER [flags]</h4>
            <p>Generate a comprehensive analysis report for any ticker.</p>
            <p><strong>Examples:</strong></p>
            <ul>
                <li><code>/report AAPL</code> - Basic report for Apple</li>
                <li><code>/report TSLA -email</code> - Report sent to email</li>
                <li><code>/report BTCUSDT -indicators=RSI,MACD -period=6mo</code> - Custom indicators and period</li>
                <li><code>/report MSFT -interval=1h -provider=yf</code> - Hourly data from Yahoo Finance</li>
            </ul>

            <h4>JSON Configuration (Advanced)</h4>
            <p><code>/report -config=JSON_STRING</code></p>
            <p><strong>Examples:</strong></p>
            <ul>
                <li><code>/report -config='{"report_type":"analysis","tickers":["AAPL","MSFT"],"period":"1y","indicators":["RSI","MACD"],"email":true}'</code></li>
                <li><code>/report -config='{"report_type":"analysis","tickers":["TSLA"],"period":"6mo","interval":"1h","indicators":["RSI","MACD","BollingerBands"],"include_fundamentals":false}'</code></li>
                <li><code>/report -config='{"report_type":"analysis","tickers":["BTCUSDT","ETHUSDT"],"period":"3mo","interval":"4h","indicators":["RSI","MACD","BollingerBands"],"include_fundamentals":false,"email":true}'</code></li>
            </ul>

            <h4>JSON Configuration Schema</h4>
            <p><strong>Supported Fields:</strong></p>
            <ul>
                <li><code>report_type</code> - "analysis", "screener", or "custom"</li>
                <li><code>tickers</code> - Array of ticker symbols ["AAPL", "MSFT"]</li>
                <li><code>period</code> - Data period: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"</li>
                <li><code>interval</code> - Data interval: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"</li>
                <li><code>provider</code> - Data provider: "yf", "alpha_vantage", "polygon"</li>
                <li><code>indicators</code> - Array of technical indicators: ["RSI", "MACD", "BollingerBands", "SMA", "EMA", "ADX", "ATR", "Stochastic", "WilliamsR"]</li>
                <li><code>fundamental_indicators</code> - Array of fundamental indicators: ["PE", "PB", "ROE", "ROA", "DebtEquity", "CurrentRatio", "EPS", "Revenue", "ProfitMargin"]</li>
                <li><code>email</code> - Boolean: true/false to send to email</li>
                <li><code>include_chart</code> - Boolean: true/false to include charts</li>
                <li><code>include_fundamentals</code> - Boolean: true/false to include fundamental analysis</li>
                <li><code>include_technicals</code> - Boolean: true/false to include technical analysis</li>
            </ul>
            <p><strong>Flags:</strong></p>
            <ul>
                <li><code>-email</code> - Send report to your verified email</li>
                <li><code>-indicators=RSI,MACD,BollingerBands</code> - Specify technical indicators</li>
                <li><code>-period=1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max</code> - Data period</li>
                <li><code>-interval=1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo</code> - Data interval</li>
                <li><code>-provider=yf,alpha_vantage,polygon</code> - Data provider</li>
                <li><code>-config=JSON_STRING</code> - Use JSON configuration for advanced options</li>
            </ul>

            <h4>Supported Indicators</h4>
            <p><strong>Technical Indicators:</strong> RSI, MACD, BollingerBands, SMA, EMA, ADX, ATR, Stochastic, WilliamsR</p>
            <p><strong>Fundamental Indicators:</strong> PE, PB, ROE, ROA, DebtEquity, CurrentRatio, EPS, Revenue, ProfitMargin</p>
        </div>

        <h3>🚨 Alert Commands</h3>
        <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h4>Price Alerts</h4>
            <p><code>/alerts add TICKER PRICE CONDITION [flags]</code></p>
            <p><strong>Examples:</strong></p>
            <ul>
                <li><code>/alerts add AAPL 150.00 above -email</code> - Alert when AAPL goes above $150</li>
                <li><code>/alerts add BTCUSDT 50000 below</code> - Alert when BTC drops below $50k</li>
            </ul>

            <h4>Indicator Alerts (Advanced)</h4>
            <p><code>/alerts add_indicator TICKER CONFIG_JSON [flags]</code></p>
            <p><strong>Examples:</strong></p>
            <ul>
                <li><code>/alerts add_indicator AAPL '{"type":"indicator","indicator":"RSI","parameters":{"period":14},"condition":{"operator":"<","value":30},"alert_action":"BUY","timeframe":"15m"}' -email</code></li>
                <li><code>/alerts add_indicator TSLA '{"type":"indicator","indicator":"BollingerBands","parameters":{"period":20},"condition":{"operator":"below_lower_band"},"alert_action":"BUY"}'</code></li>
            </ul>

            <h4>Alert Management</h4>
            <ul>
                <li><code>/alerts</code> - List all your alerts</li>
                <li><code>/alerts edit ALERT_ID [PRICE] [CONDITION] [flags]</code> - Edit alert</li>
                <li><code>/alerts delete ALERT_ID</code> - Delete alert</li>
                <li><code>/alerts pause ALERT_ID</code> - Pause alert</li>
                <li><code>/alerts resume ALERT_ID</code> - Resume alert</li>
            </ul>

            <p><strong>Alert Flags:</strong></p>
            <ul>
                <li><code>-email</code> - Send alert to email</li>
                <li><code>-timeframe=15m</code> - Set timeframe (5m, 15m, 1h, 4h, 1d)</li>
                <li><code>-action_type=BUY</code> - Set action (BUY, SELL, HOLD, notify)</li>
            </ul>
        </div>

        <h3>⏰ Schedule Commands</h3>
        <div style="background: #d1ecf1; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h4>Simple Schedules</h4>
            <p><code>/schedules add TICKER TIME [flags]</code></p>
            <p><strong>Examples:</strong></p>
            <ul>
                <li><code>/schedules add AAPL 09:00 -email</code> - Daily report at 9 AM</li>
                <li><code>/schedules add TSLA 16:30 -indicators=RSI,MACD</code> - Report with indicators at 4:30 PM</li>
            </ul>

            <h4>JSON Schedules (Advanced)</h4>
            <p><code>/schedules add_json CONFIG_JSON</code></p>
            <p><strong>Examples:</strong></p>
            <ul>
                <li><code>/schedules add_json '{"type":"report","ticker":"AAPL","scheduled_time":"09:00","period":"1y","interval":"1d","email":true}'</code></li>
                <li><code>/schedules add_json '{"type":"screener","list_type":"us_small_cap","scheduled_time":"08:00","period":"1y","interval":"1d","indicators":"PE,PB,ROE","email":true}'</code></li>
            </ul>

            <h4>Screener Schedules</h4>
            <p><code>/schedules screener LIST_TYPE [TIME] [flags]</code></p>
            <p><strong>Examples:</strong></p>
            <ul>
                <li><code>/schedules screener us_small_cap 09:00 -email</code> - Small cap screener at 9 AM</li>
                <li><code>/schedules screener us_large_cap -indicators=PE,PB,ROE</code> - Large cap screener with specific indicators</li>
            </ul>

            <h4>Schedule Management</h4>
            <ul>
                <li><code>/schedules</code> - List all your schedules</li>
                <li><code>/schedules edit SCHEDULE_ID [TIME] [flags]</code> - Edit schedule</li>
                <li><code>/schedules delete SCHEDULE_ID</code> - Delete schedule</li>
                <li><code>/schedules pause SCHEDULE_ID</code> - Pause schedule</li>
                <li><code>/schedules resume SCHEDULE_ID</code> - Resume schedule</li>
            </ul>

            <p><strong>Supported List Types:</strong> us_small_cap, us_medium_cap, us_large_cap, swiss_shares, custom_list</p>
        </div>

        <h3>🔧 Utility Commands</h3>
        <div style="background: #f8d7da; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h4>Account Management</h4>
            <ul>
                <li><code>/start</code> - Start the bot and get welcome message</li>
                <li><code>/help</code> - Show this help message</li>
                <li><code>/register your@email.com</code> - Register your email</li>
                <li><code>/verify CODE</code> - Verify email with received code</li>
                <li><code>/request_approval</code> - Request admin approval</li>
                <li><code>/info</code> - Show your account information</li>
            </ul>

            <h4>Admin Commands (Admin Only)</h4>
            <ul>
                <li><code>/admin users</code> - List all users</li>
                <li><code>/admin approve USER_ID</code> - Approve user</li>
                <li><code>/admin reject USER_ID</code> - Reject user</li>
                <li><code>/admin broadcast MESSAGE</code> - Send message to all users</li>
                <li><code>/admin setlimit USER_ID TYPE VALUE</code> - Set user limits</li>
            </ul>
        </div>

        <h3>📈 Technical Indicators</h3>
        <div style="background: #e2e3e5; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h4>Available Indicators</h4>
            <ul>
                <li><strong>RSI (Relative Strength Index)</strong> - Momentum oscillator (0-100)</li>
                <li><strong>MACD</strong> - Moving Average Convergence Divergence</li>
                <li><strong>Bollinger Bands</strong> - Volatility bands around moving average</li>
                <li><strong>SMA (Simple Moving Average)</strong> - Average price over period</li>
            </ul>

            <h4>Fundamental Indicators (Screener)</h4>
            <ul>
                <li><strong>P/E (Price-to-Earnings)</strong> - Valuation ratio</li>
                <li><strong>P/B (Price-to-Book)</strong> - Asset valuation</li>
                <li><strong>ROE (Return on Equity)</strong> - Profitability measure</li>
                <li><strong>ROA (Return on Assets)</strong> - Asset efficiency</li>
                <li><strong>Debt/Equity</strong> - Financial leverage</li>
                <li><strong>Current Ratio</strong> - Liquidity measure</li>
            </ul>
        </div>

        <h3>⚙️ Advanced Features</h3>
        <div style="background: #d4edda; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h4>JSON Configuration</h4>
            <p>For advanced users, you can use JSON configurations for complex reports, alerts and schedules:</p>

            <h5>Report Configuration Examples:</h5>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;">
{
  "report_type": "analysis",
  "tickers": ["AAPL", "MSFT"],
  "period": "1y",
  "interval": "1d",
  "provider": "yf",
  "indicators": ["RSI", "MACD"],
  "fundamental_indicators": ["PE", "PB", "ROE"],
  "email": true,
  "include_chart": true,
  "include_fundamentals": true,
  "include_technicals": true
}</pre>

            <h5>Technical Analysis Report:</h5>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;">
{
  "report_type": "analysis",
  "tickers": ["TSLA"],
  "period": "6mo",
  "interval": "1h",
  "provider": "yf",
  "indicators": ["RSI", "MACD", "BollingerBands", "SMA"],
  "include_fundamentals": false,
  "email": false
}</pre>

            <h5>Indicator Alert Examples:</h5>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;">
{
  "type": "indicator",
  "indicator": "RSI",
  "parameters": {"period": 14},
  "condition": {"operator": "<", "value": 30},
  "alert_action": "BUY",
  "timeframe": "15m"
}</pre>

            <h5>Complex Alert with Multiple Conditions:</h5>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto;">
{
  "type": "indicator",
  "logic": "AND",
  "conditions": [
    {
      "indicator": "RSI",
      "parameters": {"period": 14},
      "condition": {"operator": "<", "value": 30}
    },
    {
      "indicator": "BollingerBands",
      "parameters": {"period": 20},
      "condition": {"operator": "below_lower_band"}
    }
  ],
  "alert_action": "BUY",
  "timeframe": "15m"
}</pre>
        </div>

        <h3>📞 Support</h3>
        <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <p>If you need help or have questions:</p>
            <ul>
                <li>Check this help page for command examples</li>
                <li>Use <code>/help</code> command in the bot</li>
                <li>Contact the admin for account issues</li>
                <li>Report bugs or request features via admin panel</li>
            </ul>
        </div>

        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <p><strong>🎯 Pro Tips:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>Use <code>-email</code> flag to get reports and alerts in your email</li>
                <li>Combine multiple indicators for better analysis</li>
                <li>Set up daily schedules for regular market monitoring</li>
                <li>Use fundamental screener for undervalued stock discovery</li>
                <li>Configure alerts with specific timeframes for better timing</li>
            </ul>
        </div>
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        _logger.exception("Error in help page: ")
        return f"Error: {str(e)}", 500

@app.route('/feedback')
@login_required
def feedback():
    """Feedback management page."""
    try:
        db.init_db()
        feedback_list = db.list_feedback()

        # Apply filters
        filter_type = request.args.get('filter', '')
        if filter_type == 'open':
            feedback_list = [f for f in feedback_list if f['status'] == 'open']

        filter_info = ""
        if filter_type:
            filter_info = f" (Filtered: {filter_type.title()})"

        content = f"""
        <h3>Feedback & Feature Requests{filter_info}</h3>

        <div style="margin-bottom: 20px;">
            <a href="/feedback" class="btn btn-secondary">All Feedback</a>
            <a href="/feedback?filter=open" class="btn btn-secondary">Open Only</a>
        </div>

        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>User ID</th>
                    <th>Type</th>
                    <th>Message</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
        """

        for item in feedback_list:
            type_badge = "💬 Feedback" if item['type'] == 'feedback' else "✨ Feature"
            status_color = {"open": "🔵", "in_progress": "🟡", "closed": "🟢"}.get(item['status'], "⚪")

            content += f"""
                <tr>
                    <td>#{item['id']}</td>
                    <td>{item['user_id']}</td>
                    <td>{type_badge}</td>
                    <td>{item['message'][:100]}{'...' if len(item['message']) > 100 else ''}</td>
                    <td>{status_color} {item['status']}</td>
                    <td>{item['created']}</td>
                    <td>
                        <a href="/feedback/{item['id']}/close" class="btn btn-success">Close</a>
                    </td>
                </tr>
            """

        content += """
            </tbody>
        </table>
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        _logger.exception("Error in feedback page: ")
        return f"Error: {str(e)}", 500

@app.route('/audit')
@login_required
def audit():
    """Command audit page."""
    try:
        db.init_db()

        # Get filter parameters
        user_id = request.args.get('user_id', '')
        command = request.args.get('command', '')
        success_only = request.args.get('success_only', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        period = request.args.get('period', '')
        user_type = request.args.get('user_type', '')  # 'registered', 'non_registered', or empty for all
        page = int(request.args.get('page', 1))
        limit = 50
        offset = (page - 1) * limit

        # Handle period filter
        if period == '24h':
            from datetime import datetime, timedelta
            end_date = datetime.now().isoformat()
            start_date = (datetime.now() - timedelta(hours=24)).isoformat()

                # Get audit data
        audit_list = db.get_all_command_audit(
            limit=limit,
            offset=offset,
            user_id=user_id if user_id else None,
            command=command if command else None,
            success_only=True if success_only == 'true' else None,
            start_date=start_date if start_date else None,
            end_date=end_date if end_date else None
        )

        # Apply user type filter if specified
        if user_type == 'registered':
            audit_list = [a for a in audit_list if a['is_registered_user']]
        elif user_type == 'non_registered':
            audit_list = [a for a in audit_list if not a['is_registered_user']]

        # Get audit statistics
        stats = db.get_command_audit_stats()

        content = f"""
        <h3>Command Audit Log</h3>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{stats['total_commands']}</div>
                <div>Total Commands</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['successful_commands']}</div>
                <div>Successful</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['failed_commands']}</div>
                <div>Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['registered_users']}</div>
                <div>Registered Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['non_registered_users']}</div>
                <div>Non-Registered Users</div>
                <div class="stat-links">
                    <a href="/audit?user_type=non_registered" class="stat-link">View Commands</a>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['recent_activity_24h']}</div>
                <div>Last 24h</div>
            </div>
        </div>

        <h4>Quick Filters</h4>
        <div style="margin-bottom: 15px;">
            <a href="/audit" class="btn btn-secondary">All Commands</a>
            <a href="/audit?period=24h" class="btn btn-secondary">Last 24 Hours</a>
            <a href="/audit?user_type=registered" class="btn btn-secondary">Registered Users</a>
            <a href="/audit?user_type=non_registered" class="btn btn-secondary">Non-Registered Users</a>
            <a href="/audit?success_only=false" class="btn btn-secondary">Failed Commands</a>
        </div>

        <h4>Advanced Filters</h4>
        <form method="GET" action="/audit" style="margin-bottom: 20px;">
            <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: end;">
                <div>
                    <label>User ID:</label><br>
                    <input type="text" name="user_id" value="{user_id}" placeholder="Telegram User ID">
                </div>
                <div>
                    <label>Command:</label><br>
                    <input type="text" name="command" value="{command}" placeholder="Command name">
                </div>
                <div>
                    <label>Success Only:</label><br>
                    <select name="success_only">
                        <option value="">All</option>
                        <option value="true" {'selected' if success_only == 'true' else ''}>Success Only</option>
                        <option value="false" {'selected' if success_only == 'false' else ''}>Failed Only</option>
                    </select>
                </div>
                <div>
                    <label>User Type:</label><br>
                    <select name="user_type">
                        <option value="">All Users</option>
                        <option value="registered" {'selected' if user_type == 'registered' else ''}>Registered Only</option>
                        <option value="non_registered" {'selected' if user_type == 'non_registered' else ''}>Non-Registered Only</option>
                    </select>
                </div>
                <div>
                    <label>Period:</label><br>
                    <select name="period">
                        <option value="">All Time</option>
                        <option value="24h" {'selected' if period == '24h' else ''}>Last 24 Hours</option>
                    </select>
                </div>
                <div>
                    <label>Start Date:</label><br>
                    <input type="date" name="start_date" value="{start_date}">
                </div>
                <div>
                    <label>End Date:</label><br>
                    <input type="date" name="end_date" value="{end_date}">
                </div>
                <div>
                    <button type="submit" class="btn btn-success">Filter</button>
                    <a href="/audit" class="btn btn-secondary">Clear</a>
                </div>
            </div>
        </form>

        <table class="user-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>User ID</th>
                    <th>Command</th>
                    <th>Full Message</th>
                    <th>Registered</th>
                    <th>Email</th>
                    <th>Success</th>
                    <th>Error</th>
                    <th>Response Time</th>
                    <th>Created</th>
                </tr>
            </thead>
            <tbody>
        """

        for item in audit_list:
            registered_badge = "✅" if item['is_registered_user'] else "❌"
            success_badge = "✅" if item['success'] else "❌"
            response_time = f"{item['response_time_ms']}ms" if item['response_time_ms'] else "N/A"
            error_msg = item['error_message'][:50] + "..." if item['error_message'] and len(item['error_message']) > 50 else item['error_message'] or ""
            full_msg = item['full_message'][:100] + "..." if item['full_message'] and len(item['full_message']) > 100 else item['full_message'] or ""

            content += f"""
                <tr>
                    <td>{item['id']}</td>
                    <td class="id-cell">{item['telegram_user_id']}</td>
                    <td><code>{item['command']}</code></td>
                    <td title="{item['full_message']}">{full_msg}</td>
                    <td class="status-cell">{registered_badge}</td>
                    <td>{item['user_email'] or 'N/A'}</td>
                    <td class="status-cell">{success_badge}</td>
                    <td title="{item['error_message']}">{error_msg}</td>
                    <td class="status-cell">{response_time}</td>
                    <td>{item['created']}</td>
                </tr>
            """

        content += """
            </tbody>
        </table>

        <div style="margin-top: 20px;">
            <a href="/audit?page={prev_page}" class="btn btn-secondary" {disabled_attr}>Previous</a>
            <span style="margin: 0 10px;">Page {page_num}</span>
            <a href="/audit?page={next_page}" class="btn btn-secondary">Next</a>
        </div>
        """.format(
            prev_page=page-1 if page > 1 else 1,
            next_page=page+1,
            page_num=page,
            disabled_attr='disabled' if page <= 1 else ''
        )

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        _logger.exception("Error in audit page: ")
        return f"Error: {str(e)}", 500

@app.route('/audit/user/<user_id>')
@login_required
def user_audit(user_id):
    """User-specific command audit page."""
    try:
        db.init_db()

        # Get user command history
        user_history = db.get_user_command_history(user_id, limit=100)

        # Get user info
        user_info = db.get_user_status(user_id)
        user_email = user_info.get('email') if user_info else None
        is_registered = user_info is not None

        content = f"""
        <h3>Command History for User {user_id}</h3>

        <div style="margin-bottom: 20px;">
            <strong>User Status:</strong> {'Registered' if is_registered else 'Non-Registered'}<br>
            <strong>Email:</strong> {user_email or 'N/A'}<br>
            <strong>Total Commands:</strong> {len(user_history)}
        </div>

        <table class="user-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Command</th>
                    <th>Full Message</th>
                    <th>Success</th>
                    <th>Error</th>
                    <th>Response Time</th>
                    <th>Created</th>
                </tr>
            </thead>
            <tbody>
        """

        for item in user_history:
            success_badge = "✅" if item['success'] else "❌"
            response_time = f"{item['response_time_ms']}ms" if item['response_time_ms'] else "N/A"
            error_msg = item['error_message'][:50] + "..." if item['error_message'] and len(item['error_message']) > 50 else item['error_message'] or ""
            full_msg = item['full_message'][:100] + "..." if item['full_message'] and len(item['full_message']) > 100 else item['full_message'] or ""

            content += f"""
                <tr>
                    <td>{item['id']}</td>
                    <td><code>{item['command']}</code></td>
                    <td title="{item['full_message']}">{full_msg}</td>
                    <td class="status-cell">{success_badge}</td>
                    <td title="{item['error_message']}">{error_msg}</td>
                    <td class="status-cell">{response_time}</td>
                    <td>{item['created']}</td>
                </tr>
            """

        content += """
            </tbody>
        </table>

        <div style="margin-top: 20px;">
            <a href="/audit" class="btn btn-secondary">Back to Audit</a>
        </div>
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        _logger.exception("Error in user audit page: ")
        return f"Error: {str(e)}", 500

@app.route('/audit/non-registered')
@login_required
def non_registered_audit():
    """Show commands from non-registered users."""
    try:
        db.init_db()

        # Get filter parameters
        command = request.args.get('command', '')
        success_only = request.args.get('success_only', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        page = int(request.args.get('page', 1))
        limit = 50
        offset = (page - 1) * limit

        # Get audit data for non-registered users only
        audit_list = db.get_all_command_audit(
            limit=limit,
            offset=offset,
            command=command if command else None,
            success_only=True if success_only == 'true' else None,
            start_date=start_date if start_date else None,
            end_date=end_date if end_date else None
        )

        # Filter for non-registered users only
        audit_list = [a for a in audit_list if not a['is_registered_user']]

        # Get audit statistics for non-registered users
        stats = db.get_command_audit_stats()
        non_registered_stats = {
            "total_commands": stats['non_registered_users'],
            "recent_activity_24h": len([a for a in audit_list if a['created'] >= (datetime.now() - timedelta(hours=24)).isoformat()])
        }

        content = f"""
        <h3>Non-Registered Users Command Audit</h3>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{non_registered_stats['total_commands']}</div>
                <div>Non-Registered Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{non_registered_stats['recent_activity_24h']}</div>
                <div>Commands (24h)</div>
            </div>
        </div>

        <h4>Filters</h4>
        <form method="GET" action="/audit/non-registered" style="margin-bottom: 20px;">
            <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: end;">
                <div>
                    <label>Command:</label><br>
                    <input type="text" name="command" value="{command}" placeholder="Command name">
                </div>
                <div>
                    <label>Success Only:</label><br>
                    <select name="success_only">
                        <option value="">All</option>
                        <option value="true" {'selected' if success_only == 'true' else ''}>Success Only</option>
                        <option value="false" {'selected' if success_only == 'false' else ''}>Failed Only</option>
                    </select>
                </div>
                <div>
                    <label>Start Date:</label><br>
                    <input type="date" name="start_date" value="{start_date}">
                </div>
                <div>
                    <label>End Date:</label><br>
                    <input type="date" name="end_date" value="{end_date}">
                </div>
                <div>
                    <button type="submit" class="btn btn-success">Filter</button>
                    <a href="/audit/non-registered" class="btn btn-secondary">Clear</a>
                </div>
            </div>
        </form>

        <table class="user-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>User ID</th>
                    <th>Command</th>
                    <th>Full Message</th>
                    <th>Success</th>
                    <th>Error</th>
                    <th>Response Time</th>
                    <th>Created</th>
                </tr>
            </thead>
            <tbody>
        """

        for item in audit_list:
            success_badge = "✅" if item['success'] else "❌"
            response_time = f"{item['response_time_ms']}ms" if item['response_time_ms'] else "N/A"
            error_msg = item['error_message'][:50] + "..." if item['error_message'] and len(item['error_message']) > 50 else item['error_message'] or ""
            full_msg = item['full_message'][:100] + "..." if item['full_message'] and len(item['full_message']) > 100 else item['full_message'] or ""

            content += f"""
                <tr>
                    <td>{item['id']}</td>
                    <td class="id-cell">{item['telegram_user_id']}</td>
                    <td><code>{item['command']}</code></td>
                    <td title="{item['full_message']}">{full_msg}</td>
                    <td class="status-cell">{success_badge}</td>
                    <td title="{item['error_message']}">{error_msg}</td>
                    <td class="status-cell">{response_time}</td>
                    <td>{item['created']}</td>
                </tr>
            """

        content += """
            </tbody>
        </table>

        <div style="margin-top: 20px;">
            <a href="/audit" class="btn btn-secondary">Back to Full Audit</a>
        </div>
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        _logger.exception("Error in non-registered audit page: ")
        return f"Error: {str(e)}", 500

@app.route('/broadcast', methods=['GET', 'POST'])
@login_required
def broadcast():
    """Broadcast message page."""
    if request.method == 'POST':
        try:
            message = request.form.get('message')
            if message:
                # Send broadcast message to all users using bot API
                sent_by = session.get('username', 'admin')
                _logger.info("Starting broadcast from admin panel: %s", sent_by)

                try:
                    success_count, total_count = asyncio.run(send_broadcast_message(message, sent_by))

                    if total_count == 0:
                        flash("No registered users found to broadcast to.")
                    elif success_count == total_count:
                        flash(f"Broadcast queued successfully for all {success_count} users!")
                    else:
                        flash(f"Broadcast queued for {success_count}/{total_count} users. Some messages may fail to deliver.")

                except Exception as e:
                    _logger.exception("Error in broadcast execution: ")
                    flash(f"Error sending broadcast: {str(e)}")

                return redirect(url_for('broadcast'))
            else:
                flash("Please provide a message to broadcast.")
        except Exception as e:
            _logger.exception("Error in broadcast route: ")
            flash(f"Error sending broadcast: {str(e)}")

    # Get user count for display
    try:
        users = db.list_users()
        user_count = len(users)
    except:
        user_count = 0

    content = f"""
    <h3>Broadcast Message</h3>

    <div style="margin-bottom: 20px;">
        <a href="/broadcast/history" class="btn btn-secondary">View Broadcast History</a>
    </div>

    <form method="POST">
        <div class="form-group">
            <label for="message">Message:</label>
            <textarea name="message" id="message" rows="6" placeholder="Enter your broadcast message here..." required></textarea>
        </div>
        <button type="submit" class="btn" onclick="return confirm('Are you sure you want to send this broadcast to {user_count} registered users?')">Send Broadcast</button>
    </form>

    <p><strong>Note:</strong> This message will be sent to all {user_count} registered users via Telegram.</p>
    <p><strong>Warning:</strong> This action cannot be undone. Please review your message carefully before sending.</p>
    <p><strong>API:</strong> Messages are sent via the bot's HTTP API and queued for delivery.</p>
    """

    return render_template_string(ADMIN_TEMPLATE, content=content)

@app.route('/broadcast/history')
@login_required
def broadcast_history():
    """Broadcast history page."""
    try:
        db.init_db()
        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM broadcast_log ORDER BY created DESC LIMIT 50")
        broadcasts = c.fetchall()
        conn.close()

        content = """
        <h3>Broadcast History</h3>

        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Message</th>
                    <th>Sent By</th>
                    <th>Success Rate</th>
                    <th>Created</th>
                </tr>
            </thead>
            <tbody>
        """

        for broadcast in broadcasts:
            success_rate = f"{broadcast[3]}/{broadcast[4]}" if broadcast[4] > 0 else "0/0"
            message_preview = broadcast[1][:100] + "..." if len(broadcast[1]) > 100 else broadcast[1]

            content += f"""
                <tr>
                    <td>#{broadcast[0]}</td>
                    <td title="{broadcast[1]}">{message_preview}</td>
                    <td>{broadcast[2]}</td>
                    <td>{success_rate}</td>
                    <td>{broadcast[5]}</td>
                </tr>
            """

        content += """
            </tbody>
        </table>
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        _logger.exception("Error in broadcast history page: ")
        return f"Error: {str(e)}", 500

@app.route('/alerts/<int:alert_id>/toggle')
@login_required
def toggle_alert(alert_id):
    """Toggle alert active status."""
    try:
        alert = db.get_alert(alert_id)
        if alert:
            new_status = not alert['active']
            db.update_alert(alert_id, active=new_status)
            flash(f"Alert #{alert_id} {'activated' if new_status else 'deactivated'}.")
        else:
            flash(f"Alert #{alert_id} not found.")
        return redirect(url_for('alerts'))
    except Exception as e:
        flash(f"Error toggling alert: {str(e)}")
        return redirect(url_for('alerts'))

@app.route('/alerts/<int:alert_id>/delete')
@login_required
def delete_alert(alert_id):
    """Delete an alert."""
    try:
        db.delete_alert(alert_id)
        flash(f"Alert #{alert_id} deleted.")
        return redirect(url_for('alerts'))
    except Exception as e:
        flash(f"Error deleting alert: {str(e)}")
        return redirect(url_for('alerts'))

@app.route('/schedules/<int:schedule_id>/toggle')
@login_required
def toggle_schedule(schedule_id):
    """Toggle schedule active status."""
    try:
        schedule = db.get_schedule(schedule_id)
        if schedule:
            new_status = not schedule['active']
            db.update_schedule(schedule_id, active=new_status)
            flash(f"Schedule #{schedule_id} {'activated' if new_status else 'deactivated'}.")
        else:
            flash(f"Schedule #{schedule_id} not found.")
        return redirect(url_for('schedules'))
    except Exception as e:
        flash(f"Error toggling schedule: {str(e)}")
        return redirect(url_for('schedules'))

@app.route('/schedules/<int:schedule_id>/delete')
@login_required
def delete_schedule(schedule_id):
    """Delete a schedule."""
    try:
        db.delete_schedule(schedule_id)
        flash(f"Schedule #{schedule_id} deleted.")
        return redirect(url_for('schedules'))
    except Exception as e:
        flash(f"Error deleting schedule: {str(e)}")
        return redirect(url_for('schedules'))

@app.route('/feedback/<int:feedback_id>/close')
@login_required
def close_feedback(feedback_id):
    """Close a feedback item."""
    try:
        db.update_feedback_status(feedback_id, 'closed')
        flash(f"Feedback #{feedback_id} closed.")
        return redirect(url_for('feedback'))
    except Exception as e:
        flash(f"Error closing feedback: {str(e)}")
        return redirect(url_for('feedback'))

if __name__ == '__main__':
    _logger.info("Starting Telegram Screener Bot Admin Panel...")

    # Check if admin credentials are configured
    if not WEBGUI_LOGIN or not WEBGUI_PASSWORD:
        _logger.error("Admin credentials not configured! Please set WEBGUI_LOGIN and WEBGUI_PASSWORD environment variables.")
        _logger.error("You can set them in config/donotshare/.env file")
        sys.exit(1)

    _logger.info("Admin panel will be available at: http://localhost:%s", WEBGUI_PORT)
    _logger.info("Login with username: %s", WEBGUI_LOGIN)
    app.run(debug=False, host='0.0.0.0', port=WEBGUI_PORT)
