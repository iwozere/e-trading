-- Fix permissions for telegram_command_audits table
-- Run this on your PostgreSQL database on the Raspberry Pi

-- Grant all privileges on the table to trading_admin
GRANT ALL PRIVILEGES ON TABLE telegram_command_audits TO trading_admin;

-- Grant usage on the sequence to trading_admin
GRANT ALL PRIVILEGES ON SEQUENCE telegram_command_audits_id_seq1 TO trading_admin;

-- Optionally, change the owner to trading_admin for consistency
ALTER TABLE telegram_command_audits OWNER TO trading_admin;
ALTER SEQUENCE telegram_command_audits_id_seq1 OWNER TO trading_admin;

-- Also fix related telegram tables for consistency
GRANT ALL PRIVILEGES ON TABLE telegram_broadcast_logs TO trading_admin;
GRANT ALL PRIVILEGES ON SEQUENCE telegram_broadcast_logs_id_seq1 TO trading_admin;
ALTER TABLE telegram_broadcast_logs OWNER TO trading_admin;
ALTER SEQUENCE telegram_broadcast_logs_id_seq1 OWNER TO trading_admin;

GRANT ALL PRIVILEGES ON TABLE telegram_feedbacks TO trading_admin;
GRANT ALL PRIVILEGES ON SEQUENCE telegram_feedbacks_id_seq1 TO trading_admin;
ALTER TABLE telegram_feedbacks OWNER TO trading_admin;
ALTER SEQUENCE telegram_feedbacks_id_seq1 OWNER TO trading_admin;

GRANT ALL PRIVILEGES ON TABLE telegram_settings TO trading_admin;
ALTER TABLE telegram_settings OWNER TO trading_admin;
