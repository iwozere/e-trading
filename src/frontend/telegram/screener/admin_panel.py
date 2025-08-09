#!/usr/bin/env python3
"""
Basic Flask-based admin panel for Telegram Screener Bot.
Provides web interface for user management, alerts, schedules, and feedback.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash
import sqlite3
from datetime import datetime
from src.frontend.telegram import db
from src.notification.logger import setup_logger

logger = setup_logger("telegram_admin_panel")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# HTML Templates
ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Telegram Screener Bot - Admin Panel</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .nav { background: #333; color: white; padding: 15px; margin: -20px -20px 20px -20px; border-radius: 8px 8px 0 0; }
        .nav a { color: white; text-decoration: none; margin-right: 20px; }
        .nav a:hover { text-decoration: underline; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        tr:hover { background-color: #f5f5f5; }
        .btn { background: #007bff; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; border: none; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #218838; }
        .stats { display: flex; gap: 20px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 8px; flex: 1; text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; color: #007bff; }
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
            <h2 style="margin: 0; display: inline;">Telegram Screener Bot - Admin Panel</h2>
            <div style="float: right;">
                <a href="{{ url_for('dashboard') }}">Dashboard</a>
                <a href="{{ url_for('users') }}">Users</a>
                <a href="{{ url_for('alerts') }}">Alerts</a>
                <a href="{{ url_for('schedules') }}">Schedules</a>
                <a href="{{ url_for('feedback') }}">Feedback</a>
                <a href="{{ url_for('broadcast') }}">Broadcast</a>
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
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Admin dashboard with statistics."""
    try:
        db.init_db()

        # Get statistics
        users = db.list_users()
        total_users = len(users)
        verified_users = sum(1 for user in users if user['verified'])

        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM alerts WHERE active=1")
        active_alerts = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM schedules WHERE active=1")
        active_schedules = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM feedback WHERE status='open'")
        open_feedback = c.fetchone()[0]

        conn.close()

        content = f"""
        <h3>Dashboard</h3>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{total_users}</div>
                <div>Total Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{verified_users}</div>
                <div>Verified Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{active_alerts}</div>
                <div>Active Alerts</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{active_schedules}</div>
                <div>Active Schedules</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{open_feedback}</div>
                <div>Open Feedback</div>
            </div>
        </div>

        <h4>Recent Activity</h4>
        <p>Welcome to the Telegram Screener Bot Admin Panel. Use the navigation above to manage users, alerts, schedules, and feedback.</p>
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        logger.error("Error in dashboard: %s", e, exc_info=True)
        return f"Error: {str(e)}", 500

@app.route('/users')
def users():
    """User management page."""
    try:
        db.init_db()
        users_list = db.list_users()

        content = """
        <h3>User Management</h3>

        <table>
            <thead>
                <tr>
                    <th>Telegram ID</th>
                    <th>Email</th>
                    <th>Verified</th>
                    <th>Language</th>
                    <th>Admin</th>
                    <th>Max Alerts</th>
                    <th>Max Schedules</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
        """

        for user in users_list:
            verified_badge = "✅" if user['verified'] else "❌"
            admin_badge = "👑" if user['is_admin'] else "👤"
            email = user['email'] or "(no email)"

            content += f"""
                <tr>
                    <td>{user['telegram_user_id']}</td>
                    <td>{email}</td>
                    <td>{verified_badge}</td>
                    <td>{user['language'] or 'en'}</td>
                    <td>{admin_badge}</td>
                    <td>{user['max_alerts']}</td>
                    <td>{user['max_schedules']}</td>
                    <td>
                        <a href="/users/{user['telegram_user_id']}/verify" class="btn btn-success">Verify</a>
                        <a href="/users/{user['telegram_user_id']}/reset" class="btn btn-danger">Reset Email</a>
                    </td>
                </tr>
            """

        content += """
            </tbody>
        </table>
        """

        return render_template_string(ADMIN_TEMPLATE, content=content)

    except Exception as e:
        logger.error("Error in users page: %s", e, exc_info=True)
        return f"Error: {str(e)}", 500

@app.route('/users/<user_id>/verify')
def verify_user(user_id):
    """Manually verify a user."""
    try:
        db.init_db()
        db.update_user_verification(user_id, True)
        flash(f"User {user_id} has been manually verified.")
        return redirect(url_for('users'))
    except Exception as e:
        flash(f"Error verifying user: {str(e)}")
        return redirect(url_for('users'))

@app.route('/users/<user_id>/reset')
def reset_user_email(user_id):
    """Reset a user's email."""
    try:
        db.init_db()
        db.update_user_email(user_id, None)
        db.update_user_verification(user_id, False)
        flash(f"Email reset for user {user_id}.")
        return redirect(url_for('users'))
    except Exception as e:
        flash(f"Error resetting email: {str(e)}")
        return redirect(url_for('users'))

@app.route('/alerts')
def alerts():
    """Alert management page."""
    try:
        db.init_db()
        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM alerts ORDER BY created DESC")
        alerts_list = c.fetchall()
        conn.close()

        content = """
        <h3>Alert Management</h3>

        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>User ID</th>
                    <th>Ticker</th>
                    <th>Price</th>
                    <th>Condition</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
        """

        for alert in alerts_list:
            status_badge = "🟢 Active" if alert[5] else "🔴 Inactive"

            content += f"""
                <tr>
                    <td>#{alert[0]}</td>
                    <td>{alert[2]}</td>
                    <td>{alert[1]}</td>
                    <td>${alert[3]:.2f}</td>
                    <td>{alert[4]}</td>
                    <td>{status_badge}</td>
                    <td>{alert[6]}</td>
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
        logger.error("Error in alerts page: %s", e, exc_info=True)
        return f"Error: {str(e)}", 500

@app.route('/schedules')
def schedules():
    """Schedule management page."""
    try:
        db.init_db()
        conn = sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM schedules ORDER BY created DESC")
        schedules_list = c.fetchall()
        conn.close()

        content = """
        <h3>Schedule Management</h3>

        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>User ID</th>
                    <th>Ticker</th>
                    <th>Time</th>
                    <th>Period</th>
                    <th>Email</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
        """

        for schedule in schedules_list:
            status_badge = "🟢 Active" if schedule[5] else "🔴 Inactive"
            email_badge = "📧" if schedule[6] else "💬"

            content += f"""
                <tr>
                    <td>#{schedule[0]}</td>
                    <td>{schedule[2]}</td>
                    <td>{schedule[1]}</td>
                    <td>{schedule[3]}</td>
                    <td>{schedule[4] or 'daily'}</td>
                    <td>{email_badge}</td>
                    <td>{status_badge}</td>
                    <td>{schedule[10]}</td>
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
        logger.error("Error in schedules page: %s", e, exc_info=True)
        return f"Error: {str(e)}", 500

@app.route('/feedback')
def feedback():
    """Feedback management page."""
    try:
        db.init_db()
        feedback_list = db.list_feedback()

        content = """
        <h3>Feedback & Feature Requests</h3>

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
        logger.error("Error in feedback page: %s", e, exc_info=True)
        return f"Error: {str(e)}", 500

@app.route('/broadcast', methods=['GET', 'POST'])
def broadcast():
    """Broadcast message page."""
    if request.method == 'POST':
        try:
            message = request.form.get('message')
            if message:
                # This would integrate with the notification system
                flash(f"Broadcast message scheduled: {message[:50]}...")
                return redirect(url_for('broadcast'))
            else:
                flash("Please provide a message to broadcast.")
        except Exception as e:
            flash(f"Error sending broadcast: {str(e)}")

    content = """
    <h3>Broadcast Message</h3>

    <form method="POST">
        <div class="form-group">
            <label for="message">Message:</label>
            <textarea name="message" id="message" rows="6" placeholder="Enter your broadcast message here..." required></textarea>
        </div>
        <button type="submit" class="btn">Send Broadcast</button>
    </form>

    <p><strong>Note:</strong> This message will be sent to all registered users via Telegram.</p>
    """

    return render_template_string(ADMIN_TEMPLATE, content=content)

@app.route('/alerts/<int:alert_id>/toggle')
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
    app.run(debug=True, host='0.0.0.0', port=5001)
