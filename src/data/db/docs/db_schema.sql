-- DROP SCHEMA public;

CREATE SCHEMA public AUTHORIZATION pg_database_owner;

COMMENT ON SCHEMA public IS 'standard public schema';

-- DROP SEQUENCE public.job_schedule_runs_id_seq;

CREATE SEQUENCE public.job_schedule_runs_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.job_schedules_id_seq;

CREATE SEQUENCE public.job_schedules_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.msg_channel_configs_id_seq;

CREATE SEQUENCE public.msg_channel_configs_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.msg_channel_health_id_seq;

CREATE SEQUENCE public.msg_channel_health_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.msg_delivery_status_id_seq;

CREATE SEQUENCE public.msg_delivery_status_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.msg_messages_id_seq;

CREATE SEQUENCE public.msg_messages_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.msg_rate_limits_id_seq;

CREATE SEQUENCE public.msg_rate_limits_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.sentiment_payloads_id_seq;

CREATE SEQUENCE public.sentiment_payloads_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.ss_ad_hoc_candidates_id_seq;

CREATE SEQUENCE public.ss_ad_hoc_candidates_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.ss_alerts_id_seq;

CREATE SEQUENCE public.ss_alerts_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.ss_deep_metrics_id_seq;

CREATE SEQUENCE public.ss_deep_metrics_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.ss_finra_short_interest_id_seq;

CREATE SEQUENCE public.ss_finra_short_interest_id_seq
	MINVALUE 0
	NO MAXVALUE
	START 0
	NO CYCLE;
-- DROP SEQUENCE public.ss_snapshot_id_seq;

CREATE SEQUENCE public.ss_snapshot_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.telegram_broadcast_logs_id_seq1;

CREATE SEQUENCE public.telegram_broadcast_logs_id_seq1
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.telegram_command_audits_id_seq1;

CREATE SEQUENCE public.telegram_command_audits_id_seq1
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.telegram_feedbacks_id_seq1;

CREATE SEQUENCE public.telegram_feedbacks_id_seq1
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.trading_bots_id_seq;

CREATE SEQUENCE public.trading_bots_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.trading_performance_metrics_id_seq;

CREATE SEQUENCE public.trading_performance_metrics_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.trading_positions_id_seq;

CREATE SEQUENCE public.trading_positions_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.trading_trades_id_seq;

CREATE SEQUENCE public.trading_trades_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.usr_auth_identities_id_seq;

CREATE SEQUENCE public.usr_auth_identities_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.usr_users_id_seq;

CREATE SEQUENCE public.usr_users_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.usr_verification_codes_id_seq;

CREATE SEQUENCE public.usr_verification_codes_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.webui_audit_logs_id_seq1;

CREATE SEQUENCE public.webui_audit_logs_id_seq1
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.webui_performance_snapshots_id_seq1;

CREATE SEQUENCE public.webui_performance_snapshots_id_seq1
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.webui_strategy_templates_id_seq1;

CREATE SEQUENCE public.webui_strategy_templates_id_seq1
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE public.webui_system_config_id_seq1;

CREATE SEQUENCE public.webui_system_config_id_seq1
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;-- public.alembic_version definition

-- Drop table

-- DROP TABLE public.alembic_version;

CREATE TABLE public.alembic_version ( version_num varchar(32) NOT NULL, CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num));


-- public.job_schedules definition

-- Drop table

-- DROP TABLE public.job_schedules;

CREATE TABLE public.job_schedules ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, user_id int4 NOT NULL, "name" varchar(255) NOT NULL, job_type varchar(50) NOT NULL, "target" varchar(255) NOT NULL, task_params jsonb DEFAULT '{}'::jsonb NOT NULL, cron varchar(100) NOT NULL, enabled bool DEFAULT true NOT NULL, next_run_at timestamptz NULL, created_at timestamptz DEFAULT now() NOT NULL, updated_at timestamptz DEFAULT now() NOT NULL, CONSTRAINT job_schedules_pkey PRIMARY KEY (id));


-- public.msg_channel_configs definition

-- Drop table

-- DROP TABLE public.msg_channel_configs;

CREATE TABLE public.msg_channel_configs ( id bigserial NOT NULL, channel varchar(50) NOT NULL, enabled bool DEFAULT true NOT NULL, config jsonb NOT NULL, rate_limit_per_minute int4 DEFAULT 60 NOT NULL, max_retries int4 DEFAULT 3 NOT NULL, timeout_seconds int4 DEFAULT 30 NOT NULL, created_at timestamptz DEFAULT now() NOT NULL, updated_at timestamptz DEFAULT now() NOT NULL, CONSTRAINT msg_channel_configs_channel_key UNIQUE (channel), CONSTRAINT msg_channel_configs_max_retries_check CHECK ((max_retries >= 0)), CONSTRAINT msg_channel_configs_pkey PRIMARY KEY (id), CONSTRAINT msg_channel_configs_rate_limit_per_minute_check CHECK ((rate_limit_per_minute > 0)), CONSTRAINT msg_channel_configs_timeout_seconds_check CHECK ((timeout_seconds > 0)));
CREATE INDEX idx_channel_configs_enabled ON public.msg_channel_configs USING btree (enabled);
CREATE INDEX idx_channel_configs_updated ON public.msg_channel_configs USING btree (updated_at);
COMMENT ON TABLE public.msg_channel_configs IS 'Channel plugin configuration and settings';

-- Table Triggers

create trigger update_msg_channel_configs_updated_at before
update
    on
    public.msg_channel_configs for each row execute function update_updated_at_column();


-- public.msg_messages definition

-- Drop table

-- DROP TABLE public.msg_messages;

CREATE TABLE public.msg_messages ( id bigserial NOT NULL, message_type varchar(50) NOT NULL, priority varchar(20) DEFAULT 'NORMAL'::character varying NOT NULL, channels _text NOT NULL, recipient_id varchar(100) NULL, template_name varchar(100) NULL, "content" jsonb NOT NULL, metadata jsonb NULL, created_at timestamptz DEFAULT now() NOT NULL, scheduled_for timestamptz DEFAULT now() NOT NULL, status varchar(20) DEFAULT 'PENDING'::character varying NOT NULL, retry_count int4 DEFAULT 0 NOT NULL, max_retries int4 DEFAULT 3 NOT NULL, last_error text NULL, processed_at timestamptz NULL, locked_by varchar(100) NULL, locked_at timestamptz NULL, CONSTRAINT msg_messages_max_retries_check CHECK ((max_retries >= 0)), CONSTRAINT msg_messages_pkey PRIMARY KEY (id), CONSTRAINT msg_messages_priority_check CHECK (((priority)::text = ANY ((ARRAY['LOW'::character varying, 'NORMAL'::character varying, 'HIGH'::character varying, 'CRITICAL'::character varying])::text[]))), CONSTRAINT msg_messages_retry_count_check CHECK ((retry_count >= 0)), CONSTRAINT msg_messages_status_check CHECK (((status)::text = ANY ((ARRAY['PENDING'::character varying, 'PROCESSING'::character varying, 'DELIVERED'::character varying, 'FAILED'::character varying, 'CANCELLED'::character varying])::text[]))));
CREATE INDEX idx_msg_messages_created ON public.msg_messages USING btree (created_at);
CREATE INDEX idx_msg_messages_locked_at ON public.msg_messages USING btree (locked_at);
CREATE INDEX idx_msg_messages_locked_by ON public.msg_messages USING btree (locked_by);
CREATE INDEX idx_msg_messages_pending_poll ON public.msg_messages USING btree (status, scheduled_for, locked_by, locked_at) WHERE ((status)::text = 'PENDING'::text);
CREATE INDEX idx_msg_messages_pending_priority ON public.msg_messages USING btree (status, scheduled_for, priority) WHERE ((status)::text = 'PENDING'::text);
CREATE INDEX idx_msg_messages_priority ON public.msg_messages USING btree (priority);
CREATE INDEX idx_msg_messages_recipient ON public.msg_messages USING btree (recipient_id);
CREATE INDEX idx_msg_messages_retry_eligible ON public.msg_messages USING btree (status, retry_count, max_retries, processed_at) WHERE ((status)::text = 'FAILED'::text);
CREATE INDEX idx_msg_messages_scheduled ON public.msg_messages USING btree (scheduled_for);
CREATE INDEX idx_msg_messages_status ON public.msg_messages USING btree (status);
CREATE INDEX idx_msg_messages_type ON public.msg_messages USING btree (message_type);
COMMENT ON TABLE public.msg_messages IS 'Notification messages queue with priority and retry support';


-- public.msg_rate_limits definition

-- Drop table

-- DROP TABLE public.msg_rate_limits;

CREATE TABLE public.msg_rate_limits ( id bigserial NOT NULL, user_id varchar(100) NOT NULL, channel varchar(50) NOT NULL, tokens int4 NOT NULL, last_refill timestamptz DEFAULT now() NOT NULL, max_tokens int4 NOT NULL, refill_rate int4 NOT NULL, created_at timestamptz DEFAULT now() NOT NULL, CONSTRAINT msg_rate_limits_max_tokens_check CHECK ((max_tokens > 0)), CONSTRAINT msg_rate_limits_pkey PRIMARY KEY (id), CONSTRAINT msg_rate_limits_refill_rate_check CHECK ((refill_rate > 0)), CONSTRAINT msg_rate_limits_tokens_check CHECK ((tokens >= 0)), CONSTRAINT unique_user_channel_rate_limit UNIQUE (user_id, channel));
CREATE INDEX idx_rate_limits_channel ON public.msg_rate_limits USING btree (channel);
CREATE INDEX idx_rate_limits_last_refill ON public.msg_rate_limits USING btree (last_refill);
CREATE INDEX idx_rate_limits_user ON public.msg_rate_limits USING btree (user_id);
COMMENT ON TABLE public.msg_rate_limits IS 'Per-user rate limiting configuration and state';


-- public.msg_system_health definition

-- Drop table

-- DROP TABLE public.msg_system_health;

CREATE TABLE public.msg_system_health ( id int8 DEFAULT nextval('msg_channel_health_id_seq'::regclass) NOT NULL, component varchar(50) NULL, status varchar(20) NOT NULL, last_success timestamptz NULL, last_failure timestamptz NULL, failure_count int4 DEFAULT 0 NOT NULL, avg_response_time_ms int4 NULL, error_message text NULL, checked_at timestamptz DEFAULT now() NOT NULL, "system" varchar(50) NOT NULL, metadata text NULL, CONSTRAINT check_system_health_status CHECK (((status)::text = ANY ((ARRAY['HEALTHY'::character varying, 'DEGRADED'::character varying, 'DOWN'::character varying, 'UNKNOWN'::character varying])::text[]))), CONSTRAINT msg_channel_health_avg_response_time_ms_check CHECK ((avg_response_time_ms >= 0)), CONSTRAINT msg_channel_health_channel_key UNIQUE (component), CONSTRAINT msg_channel_health_failure_count_check CHECK ((failure_count >= 0)), CONSTRAINT msg_channel_health_pkey PRIMARY KEY (id), CONSTRAINT msg_channel_health_status_check CHECK (((status)::text = ANY ((ARRAY['HEALTHY'::character varying, 'DEGRADED'::character varying, 'DOWN'::character varying])::text[]))));
CREATE INDEX idx_channel_health_checked ON public.msg_system_health USING btree (checked_at);
CREATE INDEX idx_channel_health_status ON public.msg_system_health USING btree (status);
CREATE INDEX idx_system_health_system ON public.msg_system_health USING btree (system);
CREATE UNIQUE INDEX idx_system_health_unique ON public.msg_system_health USING btree (system, COALESCE(component, ''::character varying));
COMMENT ON TABLE public.msg_system_health IS 'Health monitoring for notification channels';


-- public.sentiment_payloads definition

-- Drop table

-- DROP TABLE public.sentiment_payloads;

CREATE TABLE public.sentiment_payloads ( id serial4 NOT NULL, ticker text NULL, "date" date NULL, provider text NULL, payload jsonb NULL, created_at timestamp DEFAULT now() NULL, CONSTRAINT sentiment_payloads_pkey PRIMARY KEY (id));


-- public.ss_ad_hoc_candidates definition

-- Drop table

-- DROP TABLE public.ss_ad_hoc_candidates;

CREATE TABLE public.ss_ad_hoc_candidates ( id bigserial NOT NULL, ticker varchar(10) NOT NULL, reason text NULL, first_seen timestamptz DEFAULT CURRENT_TIMESTAMP NOT NULL, expires_at timestamptz NULL, active bool DEFAULT true NOT NULL, promoted_by_screener bool DEFAULT false NOT NULL, CONSTRAINT ss_ad_hoc_candidates_pkey PRIMARY KEY (id), CONSTRAINT ss_ad_hoc_candidates_ticker_key UNIQUE (ticker));
CREATE INDEX idx_ss_adhoc_active ON public.ss_ad_hoc_candidates USING btree (active, expires_at);
CREATE INDEX idx_ss_adhoc_expires_at ON public.ss_ad_hoc_candidates USING btree (expires_at);
CREATE INDEX idx_ss_adhoc_promoted ON public.ss_ad_hoc_candidates USING btree (promoted_by_screener, active);


-- public.ss_alerts definition

-- Drop table

-- DROP TABLE public.ss_alerts;

CREATE TABLE public.ss_alerts ( id bigserial NOT NULL, ticker varchar(10) NOT NULL, alert_level varchar(10) NOT NULL, reason text NULL, squeeze_score numeric(5, 4) NULL, "timestamp" timestamptz DEFAULT CURRENT_TIMESTAMP NOT NULL, sent bool DEFAULT false NOT NULL, cooldown_expires timestamptz NULL, notification_id varchar(50) NULL, CONSTRAINT ss_alerts_alert_level_check CHECK (((alert_level)::text = ANY ((ARRAY['LOW'::character varying, 'MEDIUM'::character varying, 'HIGH'::character varying])::text[]))), CONSTRAINT ss_alerts_pkey PRIMARY KEY (id), CONSTRAINT ss_alerts_squeeze_score_check CHECK (((squeeze_score >= (0)::numeric) AND (squeeze_score <= (1)::numeric))));
CREATE INDEX idx_ss_alerts_alert_level_timestamp ON public.ss_alerts USING btree (alert_level, "timestamp" DESC);
CREATE INDEX idx_ss_alerts_sent_timestamp ON public.ss_alerts USING btree (sent, "timestamp" DESC);
CREATE INDEX idx_ss_alerts_ticker_cooldown ON public.ss_alerts USING btree (ticker, cooldown_expires);
CREATE INDEX idx_ss_alerts_timestamp_desc ON public.ss_alerts USING btree ("timestamp" DESC);


-- public.ss_deep_metrics definition

-- Drop table

-- DROP TABLE public.ss_deep_metrics;

CREATE TABLE public.ss_deep_metrics ( id bigserial NOT NULL, ticker varchar(10) NOT NULL, "date" date NOT NULL, volume_spike numeric(6, 2) NULL, call_put_ratio numeric(6, 2) NULL, sentiment_24h numeric(4, 3) NULL, borrow_fee_pct numeric(5, 4) NULL, squeeze_score numeric(5, 4) NULL, alert_level varchar(10) NULL, raw_payload jsonb NULL, created_at timestamptz DEFAULT CURRENT_TIMESTAMP NOT NULL, sentiment_score_24h float4 NULL, mentions_24h int4 NULL, unique_authors_24h int4 NULL, mentions_growth_7d float4 NULL, positive_ratio_24h float4 NULL, virality_index float4 NULL, bot_pct float4 NULL, sentiment_raw_payload jsonb NULL, sentiment_data_quality jsonb NULL, CONSTRAINT ss_deep_metrics_alert_level_check CHECK (((alert_level)::text = ANY ((ARRAY['LOW'::character varying, 'MEDIUM'::character varying, 'HIGH'::character varying])::text[]))), CONSTRAINT ss_deep_metrics_borrow_fee_pct_check CHECK ((borrow_fee_pct >= (0)::numeric)), CONSTRAINT ss_deep_metrics_call_put_ratio_check CHECK ((call_put_ratio >= (0)::numeric)), CONSTRAINT ss_deep_metrics_pkey PRIMARY KEY (id), CONSTRAINT ss_deep_metrics_sentiment_24h_check CHECK (((sentiment_24h >= ('-1'::integer)::numeric) AND (sentiment_24h <= (1)::numeric))), CONSTRAINT ss_deep_metrics_squeeze_score_check CHECK (((squeeze_score >= (0)::numeric) AND (squeeze_score <= (1)::numeric))), CONSTRAINT ss_deep_metrics_ticker_date_key UNIQUE (ticker, date), CONSTRAINT ss_deep_metrics_volume_spike_check CHECK ((volume_spike >= (0)::numeric)));
CREATE INDEX idx_ss_deep_metrics_alert_level ON public.ss_deep_metrics USING btree (alert_level, date DESC);
CREATE INDEX idx_ss_deep_metrics_created_at ON public.ss_deep_metrics USING btree (created_at);
CREATE INDEX idx_ss_deep_metrics_date_desc ON public.ss_deep_metrics USING btree (date DESC);
CREATE INDEX idx_ss_deep_metrics_sentiment24 ON public.ss_deep_metrics USING btree (sentiment_24h);
CREATE INDEX idx_ss_deep_metrics_squeeze_score_desc ON public.ss_deep_metrics USING btree (squeeze_score DESC, date DESC);
CREATE INDEX idx_ss_deep_metrics_ticker_date ON public.ss_deep_metrics USING btree (ticker, date);


-- public.ss_snapshot definition

-- Drop table

-- DROP TABLE public.ss_snapshot;

CREATE TABLE public.ss_snapshot ( id bigserial NOT NULL, ticker varchar(10) NOT NULL, run_date date NOT NULL, short_interest_pct numeric(5, 4) NULL, days_to_cover numeric(8, 2) NULL, float_shares int8 NULL, avg_volume_14d int8 NULL, market_cap int8 NULL, screener_score numeric(5, 4) NULL, raw_payload jsonb NULL, data_quality numeric(3, 2) NULL, created_at timestamptz DEFAULT CURRENT_TIMESTAMP NOT NULL, CONSTRAINT ss_snapshot_data_quality_check CHECK (((data_quality >= (0)::numeric) AND (data_quality <= (1)::numeric))), CONSTRAINT ss_snapshot_days_to_cover_check CHECK ((days_to_cover >= (0)::numeric)), CONSTRAINT ss_snapshot_pkey PRIMARY KEY (id), CONSTRAINT ss_snapshot_screener_score_check CHECK (((screener_score >= (0)::numeric) AND (screener_score <= (1)::numeric))), CONSTRAINT ss_snapshot_short_interest_pct_check CHECK (((short_interest_pct >= (0)::numeric) AND (short_interest_pct <= (1)::numeric))));
CREATE INDEX idx_ss_snapshot_created_at ON public.ss_snapshot USING btree (created_at);
CREATE INDEX idx_ss_snapshot_run_date_desc ON public.ss_snapshot USING btree (run_date DESC);
CREATE INDEX idx_ss_snapshot_screener_score_desc ON public.ss_snapshot USING btree (screener_score DESC, run_date DESC);
CREATE INDEX idx_ss_snapshot_ticker_date ON public.ss_snapshot USING btree (ticker, run_date);


-- public.telegram_broadcast_logs definition

-- Drop table

-- DROP TABLE public.telegram_broadcast_logs;

CREATE TABLE public.telegram_broadcast_logs ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, message text NOT NULL, sent_by varchar(255) NOT NULL, success_count int4 NULL, total_count int4 NULL, created_at timestamptz DEFAULT now() NULL, CONSTRAINT telegram_broadcast_logs_pkey PRIMARY KEY (id));


-- public.telegram_command_audits definition

-- Drop table

-- DROP TABLE public.telegram_command_audits;

CREATE TABLE public.telegram_command_audits ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, telegram_user_id varchar(255) NOT NULL, command varchar(255) NOT NULL, full_message text NULL, is_registered_user bool NULL, user_email varchar(255) NULL, success bool NULL, error_message text NULL, response_time_ms int4 NULL, created_at timestamptz DEFAULT now() NULL, CONSTRAINT telegram_command_audits_pkey PRIMARY KEY (id));
CREATE INDEX ix_telegram_command_audits_command ON public.telegram_command_audits USING btree (command);
CREATE INDEX ix_telegram_command_audits_created ON public.telegram_command_audits USING btree (created_at);
CREATE INDEX ix_telegram_command_audits_success ON public.telegram_command_audits USING btree (success);
CREATE INDEX ix_telegram_command_audits_telegram_user_id ON public.telegram_command_audits USING btree (telegram_user_id);


-- public.telegram_settings definition

-- Drop table

-- DROP TABLE public.telegram_settings;

CREATE TABLE public.telegram_settings ( "key" varchar(100) NOT NULL, value text NULL, CONSTRAINT telegram_settings_pkey PRIMARY KEY (key));


-- public.usr_users definition

-- Drop table

-- DROP TABLE public.usr_users;

CREATE TABLE public.usr_users ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, email varchar(100) NULL, "role" varchar(20) DEFAULT 'trader'::character varying NOT NULL, is_active bool DEFAULT true NULL, created_at timestamptz DEFAULT now() NULL, updated_at timestamp NULL, last_login timestamp NULL, CONSTRAINT users_pkey PRIMARY KEY (id));
CREATE INDEX ix_users_email ON public.usr_users USING btree (email);
COMMENT ON TABLE public.usr_users IS 'User accounts and authentication';


-- public.webui_performance_snapshots definition

-- Drop table

-- DROP TABLE public.webui_performance_snapshots;

CREATE TABLE public.webui_performance_snapshots ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, strategy_id varchar(100) NOT NULL, "timestamp" timestamptz DEFAULT now() NULL, pnl jsonb NOT NULL, positions jsonb NULL, trades_count int4 DEFAULT 0 NULL, win_rate jsonb NULL, drawdown jsonb NULL, metrics jsonb NULL, CONSTRAINT webui_performance_snapshots_pkey PRIMARY KEY (id));
CREATE INDEX ix_webui_performance_snapshots_strategy_id ON public.webui_performance_snapshots USING btree (strategy_id);


-- public.webui_system_config definition

-- Drop table

-- DROP TABLE public.webui_system_config;

CREATE TABLE public.webui_system_config ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, "key" varchar(100) NOT NULL, value jsonb NOT NULL, description text NULL, created_at timestamptz DEFAULT now() NULL, updated_at timestamp NULL, CONSTRAINT webui_system_config_pkey PRIMARY KEY (id));


-- public.job_schedule_runs definition

-- Drop table

-- DROP TABLE public.job_schedule_runs;

CREATE TABLE public.job_schedule_runs ( job_type text NOT NULL, job_id int4 NULL, user_id int8 NULL, status text NULL, scheduled_for timestamptz NULL, enqueued_at timestamptz DEFAULT now() NULL, started_at timestamptz NULL, finished_at timestamptz NULL, job_snapshot jsonb NULL, "result" jsonb NULL, "error" text NULL, id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, CONSTRAINT job_schedule_runs_pkey PRIMARY KEY (id), CONSTRAINT job_schedule_runs_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.job_schedules(id) ON DELETE CASCADE);
CREATE UNIQUE INDEX ux_runs_job_scheduled_for ON public.job_schedule_runs USING btree (job_type, job_id, scheduled_for);


-- public.msg_delivery_status definition

-- Drop table

-- DROP TABLE public.msg_delivery_status;

CREATE TABLE public.msg_delivery_status ( id bigserial NOT NULL, message_id int8 NOT NULL, channel varchar(50) NOT NULL, status varchar(20) NOT NULL, delivered_at timestamptz NULL, response_time_ms int4 NULL, error_message text NULL, external_id varchar(255) NULL, created_at timestamptz DEFAULT now() NOT NULL, CONSTRAINT msg_delivery_status_pkey PRIMARY KEY (id), CONSTRAINT msg_delivery_status_response_time_ms_check CHECK ((response_time_ms >= 0)), CONSTRAINT msg_delivery_status_status_check CHECK (((status)::text = ANY ((ARRAY['PENDING'::character varying, 'SENT'::character varying, 'DELIVERED'::character varying, 'FAILED'::character varying, 'BOUNCED'::character varying])::text[]))), CONSTRAINT msg_delivery_status_message_id_fkey FOREIGN KEY (message_id) REFERENCES public.msg_messages(id) ON DELETE CASCADE);
CREATE INDEX idx_delivery_status_channel ON public.msg_delivery_status USING btree (channel);
CREATE INDEX idx_delivery_status_composite ON public.msg_delivery_status USING btree (message_id, channel, status);
CREATE INDEX idx_delivery_status_delivered ON public.msg_delivery_status USING btree (delivered_at);
CREATE INDEX idx_delivery_status_message ON public.msg_delivery_status USING btree (message_id);
CREATE INDEX idx_delivery_status_status ON public.msg_delivery_status USING btree (status);
COMMENT ON TABLE public.msg_delivery_status IS 'Per-channel delivery status tracking for each message';


-- public.telegram_feedbacks definition

-- Drop table

-- DROP TABLE public.telegram_feedbacks;

CREATE TABLE public.telegram_feedbacks ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, user_id int4 NOT NULL, "type" varchar(50) NULL, message text NULL, created_at timestamptz DEFAULT now() NULL, status varchar(20) NULL, CONSTRAINT telegram_feedbacks_pkey PRIMARY KEY (id), CONSTRAINT telegram_feedbacks_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usr_users(id) ON DELETE CASCADE);


-- public.trading_bots definition

-- Drop table

-- DROP TABLE public.trading_bots;

CREATE TABLE public.trading_bots ( "type" varchar(20) NOT NULL, status varchar(20) NOT NULL, started_at timestamp NULL, last_heartbeat timestamp NULL, error_count int4 NULL, current_balance numeric(20, 8) NULL, total_pnl numeric(20, 8) NULL, extra_metadata jsonb NULL, created_at timestamptz DEFAULT now() NULL, updated_at timestamp NULL, config jsonb NOT NULL, user_id int4 NOT NULL, id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, description text NULL, CONSTRAINT trading_bots_pk PRIMARY KEY (id), CONSTRAINT trading_bots_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usr_users(id) ON DELETE CASCADE);

-- Column comments

COMMENT ON COLUMN public.trading_bots.description IS 'Trading bot description. Defined by the trader, who creates the bot.';

-- Table Triggers

create trigger update_bots_updated_at before
update
    on
    public.trading_bots for each row execute function update_updated_at_column();


-- public.trading_performance_metrics definition

-- Drop table

-- DROP TABLE public.trading_performance_metrics;

CREATE TABLE public.trading_performance_metrics ( bot_id int4 NOT NULL, trade_type varchar(10) NOT NULL, symbol varchar(20) NULL, "interval" varchar(10) NULL, entry_logic_name varchar(100) NULL, exit_logic_name varchar(100) NULL, metrics jsonb NOT NULL, calculated_at timestamp NULL, created_at timestamptz DEFAULT now() NULL, id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, CONSTRAINT trading_performance_metrics_pkey PRIMARY KEY (id), CONSTRAINT trading_perf_metrics_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.trading_bots(id) ON DELETE CASCADE);
CREATE INDEX ix_trading_performance_metrics_bot_id ON public.trading_performance_metrics USING btree (bot_id);
CREATE INDEX ix_trading_performance_metrics_calculated_at ON public.trading_performance_metrics USING btree (calculated_at);
CREATE INDEX ix_trading_performance_metrics_symbol ON public.trading_performance_metrics USING btree (symbol);
COMMENT ON TABLE public.trading_performance_metrics IS 'Performance metrics for trading strategies';


-- public.trading_positions definition

-- Drop table

-- DROP TABLE public.trading_positions;

CREATE TABLE public.trading_positions ( bot_id int4 NOT NULL, trade_type varchar(10) NOT NULL, symbol varchar(20) NOT NULL, direction varchar(10) NOT NULL, opened_at timestamp NULL, closed_at timestamp NULL, qty_open numeric(20, 8) DEFAULT 0 NOT NULL, avg_price numeric(20, 8) NULL, realized_pnl numeric(20, 8) DEFAULT 0 NULL, status varchar(12) NOT NULL, extra_metadata jsonb NULL, id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, CONSTRAINT trading_positions_pkey PRIMARY KEY (id), CONSTRAINT trading_positions_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.trading_bots(id) ON DELETE CASCADE);
CREATE INDEX ix_trading_positions_bot_id ON public.trading_positions USING btree (bot_id);
CREATE INDEX ix_trading_positions_symbol ON public.trading_positions USING btree (symbol);
COMMENT ON TABLE public.trading_positions IS 'Open and closed trading positions';


-- public.trading_trades definition

-- Drop table

-- DROP TABLE public.trading_trades;

CREATE TABLE public.trading_trades ( bot_id int4 NOT NULL, trade_type varchar(10) NOT NULL, strategy_name varchar(100) NULL, entry_logic_name varchar(100) NOT NULL, exit_logic_name varchar(100) NOT NULL, symbol varchar(20) NOT NULL, "interval" varchar(10) NOT NULL, entry_time timestamp NULL, exit_time timestamp NULL, buy_order_created timestamp NULL, buy_order_closed timestamp NULL, sell_order_created timestamp NULL, sell_order_closed timestamp NULL, entry_price numeric(20, 8) NULL, exit_price numeric(20, 8) NULL, entry_value numeric(20, 8) NULL, exit_value numeric(20, 8) NULL, "size" numeric(20, 8) NULL, direction varchar(10) NOT NULL, commission numeric(20, 8) NULL, gross_pnl numeric(20, 8) NULL, net_pnl numeric(20, 8) NULL, pnl_percentage numeric(10, 4) NULL, exit_reason varchar(100) NULL, status varchar(20) NOT NULL, extra_metadata jsonb NULL, created_at timestamptz DEFAULT now() NULL, updated_at timestamp NULL, position_id int4 NULL, id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, CONSTRAINT trading_trades_pkey PRIMARY KEY (id), CONSTRAINT trading_trades_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.trading_bots(id) ON DELETE CASCADE, CONSTRAINT trading_trades_position_id_fkey FOREIGN KEY (position_id) REFERENCES public.trading_positions(id) ON DELETE SET NULL);
CREATE INDEX ix_trading_trades_bot_id ON public.trading_trades USING btree (bot_id);
CREATE INDEX ix_trading_trades_entry_time ON public.trading_trades USING btree (entry_time);
CREATE INDEX ix_trading_trades_status ON public.trading_trades USING btree (status);
CREATE INDEX ix_trading_trades_strategy_name ON public.trading_trades USING btree (strategy_name);
CREATE INDEX ix_trading_trades_symbol ON public.trading_trades USING btree (symbol);
CREATE INDEX ix_trading_trades_trade_type ON public.trading_trades USING btree (trade_type);
COMMENT ON TABLE public.trading_trades IS 'Individual trade records';


-- public.usr_auth_identities definition

-- Drop table

-- DROP TABLE public.usr_auth_identities;

CREATE TABLE public.usr_auth_identities ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, user_id int4 NOT NULL, provider varchar(32) NOT NULL, external_id varchar(255) NOT NULL, metadata jsonb NULL, created_at timestamptz DEFAULT now() NULL, CONSTRAINT auth_identities_pkey PRIMARY KEY (id), CONSTRAINT auth_identities_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usr_users(id) ON DELETE CASCADE);
CREATE INDEX ix_auth_identities_provider ON public.usr_auth_identities USING btree (provider);
CREATE INDEX ix_auth_identities_user_id ON public.usr_auth_identities USING btree (user_id);


-- public.usr_verification_codes definition

-- Drop table

-- DROP TABLE public.usr_verification_codes;

CREATE TABLE public.usr_verification_codes ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, user_id int4 NOT NULL, code varchar(32) NOT NULL, sent_time int4 NOT NULL, provider varchar(20) DEFAULT 'telegram'::character varying NULL, created_at timestamptz DEFAULT now() NULL, CONSTRAINT verification_codes_pkey PRIMARY KEY (id), CONSTRAINT verification_codes_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usr_users(id) ON DELETE CASCADE);
CREATE INDEX ix_verification_codes_user_id ON public.usr_verification_codes USING btree (user_id);


-- public.webui_audit_logs definition

-- Drop table

-- DROP TABLE public.webui_audit_logs;

CREATE TABLE public.webui_audit_logs ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, user_id int4 NOT NULL, "action" varchar(100) NOT NULL, resource_type varchar(50) NULL, resource_id varchar(100) NULL, details jsonb NULL, ip_address varchar(45) NULL, user_agent varchar(500) NULL, created_at timestamptz DEFAULT now() NULL, CONSTRAINT webui_audit_logs_pkey PRIMARY KEY (id), CONSTRAINT webui_audit_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usr_users(id));
CREATE INDEX ix_webui_audit_logs_action ON public.webui_audit_logs USING btree (action);
CREATE INDEX ix_webui_audit_logs_user_id ON public.webui_audit_logs USING btree (user_id);


-- public.webui_strategy_templates definition

-- Drop table

-- DROP TABLE public.webui_strategy_templates;

CREATE TABLE public.webui_strategy_templates ( id int4 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1 NO CYCLE) NOT NULL, "name" varchar(100) NOT NULL, description text NULL, template_data jsonb NOT NULL, is_public bool DEFAULT false NULL, created_by int4 NOT NULL, created_at timestamptz DEFAULT now() NULL, updated_at timestamp NULL, CONSTRAINT webui_strategy_templates_pkey PRIMARY KEY (id), CONSTRAINT webui_strategy_templates_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.usr_users(id));
CREATE INDEX ix_webui_strategy_templates_created_by ON public.webui_strategy_templates USING btree (created_by);


-- public.v_channel_health_summary source

CREATE OR REPLACE VIEW public.v_channel_health_summary
AS SELECT ch.component AS channel,
    ch.status,
    ch.failure_count,
    ch.avg_response_time_ms,
    ch.last_success,
    ch.last_failure,
    ch.checked_at,
    cc.enabled AS config_enabled,
    cc.rate_limit_per_minute,
    cc.max_retries AS config_max_retries,
    cc.timeout_seconds
   FROM msg_system_health ch
     LEFT JOIN msg_channel_configs cc ON ch.component::text = cc.channel::text
  ORDER BY ch.component;

COMMENT ON VIEW public.v_channel_health_summary IS 'Combined channel health and configuration view';


-- public.v_delivery_stats source

CREATE OR REPLACE VIEW public.v_delivery_stats
AS SELECT channel,
    count(*) AS total_deliveries,
    count(
        CASE
            WHEN status::text = 'DELIVERED'::text THEN 1
            ELSE NULL::integer
        END) AS successful_deliveries,
    count(
        CASE
            WHEN status::text = 'FAILED'::text THEN 1
            ELSE NULL::integer
        END) AS failed_deliveries,
    count(
        CASE
            WHEN status::text = 'BOUNCED'::text THEN 1
            ELSE NULL::integer
        END) AS bounced_deliveries,
    round(avg(
        CASE
            WHEN status::text = 'DELIVERED'::text THEN response_time_ms
            ELSE NULL::integer
        END), 2) AS avg_response_time_ms,
    date_trunc('day'::text, created_at) AS delivery_date
   FROM msg_delivery_status ds
  WHERE created_at >= (now() - '30 days'::interval)
  GROUP BY channel, (date_trunc('day'::text, created_at))
  ORDER BY (date_trunc('day'::text, created_at)) DESC, channel;

COMMENT ON VIEW public.v_delivery_stats IS 'Daily delivery statistics by channel';


-- public.v_pending_messages source

CREATE OR REPLACE VIEW public.v_pending_messages
AS SELECT id,
    message_type,
    priority,
    channels,
    recipient_id,
    content,
    metadata,
    scheduled_for,
    retry_count,
    max_retries,
    created_at,
        CASE
            WHEN priority::text = 'CRITICAL'::text THEN 1
            WHEN priority::text = 'HIGH'::text THEN 2
            WHEN priority::text = 'NORMAL'::text THEN 3
            WHEN priority::text = 'LOW'::text THEN 4
            ELSE 5
        END AS priority_order
   FROM msg_messages m
  WHERE status::text = 'PENDING'::text AND scheduled_for <= now()
  ORDER BY (
        CASE
            WHEN priority::text = 'CRITICAL'::text THEN 1
            WHEN priority::text = 'HIGH'::text THEN 2
            WHEN priority::text = 'NORMAL'::text THEN 3
            WHEN priority::text = 'LOW'::text THEN 4
            ELSE 5
        END), scheduled_for;

COMMENT ON VIEW public.v_pending_messages IS 'Messages ready for processing, ordered by priority';


-- public.v_retry_eligible_messages source

CREATE OR REPLACE VIEW public.v_retry_eligible_messages
AS SELECT id,
    message_type,
    priority,
    channels,
    recipient_id,
    content,
    retry_count,
    max_retries,
    last_error,
    processed_at,
    created_at
   FROM msg_messages m
  WHERE status::text = 'FAILED'::text AND retry_count < max_retries AND (processed_at IS NULL OR processed_at <= (now() - '00:05:00'::interval))
  ORDER BY processed_at NULLS FIRST, created_at;

COMMENT ON VIEW public.v_retry_eligible_messages IS 'Failed messages eligible for retry';



-- DROP FUNCTION public.cleanup_old_messages(int4);

CREATE OR REPLACE FUNCTION public.cleanup_old_messages(p_days_to_keep integer DEFAULT 30)
 RETURNS integer
 LANGUAGE plpgsql
AS $function$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM msg_messages 
    WHERE status = 'DELIVERED' 
      AND created_at < NOW() - (p_days_to_keep || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$function$
;

-- DROP FUNCTION public.enqueue_message(varchar, _text, jsonb, varchar, varchar, varchar, jsonb, timestamptz, int4);

CREATE OR REPLACE FUNCTION public.enqueue_message(p_message_type character varying, p_channels text[], p_content jsonb, p_priority character varying DEFAULT 'NORMAL'::character varying, p_recipient_id character varying DEFAULT NULL::character varying, p_template_name character varying DEFAULT NULL::character varying, p_metadata jsonb DEFAULT NULL::jsonb, p_scheduled_for timestamp with time zone DEFAULT now(), p_max_retries integer DEFAULT 3)
 RETURNS bigint
 LANGUAGE plpgsql
AS $function$
DECLARE
    message_id BIGINT;
BEGIN
    INSERT INTO msg_messages (
        message_type, priority, channels, recipient_id, template_name,
        content, metadata, scheduled_for, max_retries
    ) VALUES (
        p_message_type, p_priority, p_channels, p_recipient_id, p_template_name,
        p_content, p_metadata, p_scheduled_for, p_max_retries
    ) RETURNING id INTO message_id;
    
    RETURN message_id;
END;
$function$
;

-- DROP FUNCTION public.get_queue_statistics();

CREATE OR REPLACE FUNCTION public.get_queue_statistics()
 RETURNS TABLE(status character varying, priority character varying, count bigint)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    SELECT 
        m.status,
        m.priority,
        COUNT(*) as count
    FROM msg_messages m
    GROUP BY m.status, m.priority
    ORDER BY m.status, 
        CASE m.priority
            WHEN 'CRITICAL' THEN 1
            WHEN 'HIGH' THEN 2
            WHEN 'NORMAL' THEN 3
            WHEN 'LOW' THEN 4
        END;
END;
$function$
;

-- DROP FUNCTION public.record_delivery_status(int8, varchar, varchar, int4, text, varchar);

CREATE OR REPLACE FUNCTION public.record_delivery_status(p_message_id bigint, p_channel character varying, p_status character varying, p_response_time_ms integer DEFAULT NULL::integer, p_error_message text DEFAULT NULL::text, p_external_id character varying DEFAULT NULL::character varying)
 RETURNS bigint
 LANGUAGE plpgsql
AS $function$
DECLARE
    delivery_id BIGINT;
BEGIN
    INSERT INTO msg_delivery_status (
        message_id, channel, status, delivered_at, response_time_ms, error_message, external_id
    ) VALUES (
        p_message_id, p_channel, p_status, 
        CASE WHEN p_status IN ('DELIVERED', 'SENT') THEN NOW() ELSE NULL END,
        p_response_time_ms, p_error_message, p_external_id
    ) RETURNING id INTO delivery_id;
    
    RETURN delivery_id;
END;
$function$
;

-- DROP FUNCTION public.update_channel_health(varchar, varchar, int4, text);

CREATE OR REPLACE FUNCTION public.update_channel_health(p_channel character varying, p_status character varying, p_response_time_ms integer DEFAULT NULL::integer, p_error_message text DEFAULT NULL::text)
 RETURNS void
 LANGUAGE plpgsql
AS $function$
BEGIN
    INSERT INTO msg_channel_health (
        channel, status, 
        last_success, last_failure, 
        failure_count, avg_response_time_ms, error_message
    ) VALUES (
        p_channel, p_status,
        CASE WHEN p_status = 'HEALTHY' THEN NOW() ELSE NULL END,
        CASE WHEN p_status IN ('DEGRADED', 'DOWN') THEN NOW() ELSE NULL END,
        CASE WHEN p_status IN ('DEGRADED', 'DOWN') THEN 1 ELSE 0 END,
        p_response_time_ms, p_error_message
    )
    ON CONFLICT (channel) DO UPDATE SET
        status = EXCLUDED.status,
        last_success = CASE 
            WHEN EXCLUDED.status = 'HEALTHY' THEN NOW() 
            ELSE msg_channel_health.last_success 
        END,
        last_failure = CASE 
            WHEN EXCLUDED.status IN ('DEGRADED', 'DOWN') THEN NOW() 
            ELSE msg_channel_health.last_failure 
        END,
        failure_count = CASE 
            WHEN EXCLUDED.status = 'HEALTHY' THEN 0
            WHEN EXCLUDED.status IN ('DEGRADED', 'DOWN') AND msg_channel_health.status = 'HEALTHY' THEN 1
            WHEN EXCLUDED.status IN ('DEGRADED', 'DOWN') THEN msg_channel_health.failure_count + 1
            ELSE msg_channel_health.failure_count
        END,
        avg_response_time_ms = CASE 
            WHEN p_response_time_ms IS NOT NULL THEN 
                COALESCE((msg_channel_health.avg_response_time_ms + p_response_time_ms) / 2, p_response_time_ms)
            ELSE msg_channel_health.avg_response_time_ms
        END,
        error_message = COALESCE(EXCLUDED.error_message, msg_channel_health.error_message),
        checked_at = NOW();
END;
$function$
;

-- DROP FUNCTION public.update_message_status(int8, varchar, text);

CREATE OR REPLACE FUNCTION public.update_message_status(p_message_id bigint, p_status character varying, p_error_message text DEFAULT NULL::text)
 RETURNS boolean
 LANGUAGE plpgsql
AS $function$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE msg_messages 
    SET 
        status = p_status,
        processed_at = NOW(),
        last_error = CASE WHEN p_error_message IS NOT NULL THEN p_error_message ELSE last_error END,
        retry_count = CASE WHEN p_status = 'PROCESSING' AND status = 'FAILED' THEN retry_count + 1 ELSE retry_count END
    WHERE id = p_message_id;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count > 0;
END;
$function$
;

-- DROP FUNCTION public.update_ss_finra_updated_at();

CREATE OR REPLACE FUNCTION public.update_ss_finra_updated_at()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$function$
;

-- DROP FUNCTION public.update_updated_at_column();

CREATE OR REPLACE FUNCTION public.update_updated_at_column()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$function$
;

-- DROP FUNCTION public.uuid_generate_v1();

CREATE OR REPLACE FUNCTION public.uuid_generate_v1()
 RETURNS uuid
 LANGUAGE c
 PARALLEL SAFE STRICT
AS '$libdir/uuid-ossp', $function$uuid_generate_v1$function$
;

-- DROP FUNCTION public.uuid_generate_v1mc();

CREATE OR REPLACE FUNCTION public.uuid_generate_v1mc()
 RETURNS uuid
 LANGUAGE c
 PARALLEL SAFE STRICT
AS '$libdir/uuid-ossp', $function$uuid_generate_v1mc$function$
;

-- DROP FUNCTION public.uuid_generate_v3(uuid, text);

CREATE OR REPLACE FUNCTION public.uuid_generate_v3(namespace uuid, name text)
 RETURNS uuid
 LANGUAGE c
 IMMUTABLE PARALLEL SAFE STRICT
AS '$libdir/uuid-ossp', $function$uuid_generate_v3$function$
;

-- DROP FUNCTION public.uuid_generate_v4();

CREATE OR REPLACE FUNCTION public.uuid_generate_v4()
 RETURNS uuid
 LANGUAGE c
 PARALLEL SAFE STRICT
AS '$libdir/uuid-ossp', $function$uuid_generate_v4$function$
;

-- DROP FUNCTION public.uuid_generate_v5(uuid, text);

CREATE OR REPLACE FUNCTION public.uuid_generate_v5(namespace uuid, name text)
 RETURNS uuid
 LANGUAGE c
 IMMUTABLE PARALLEL SAFE STRICT
AS '$libdir/uuid-ossp', $function$uuid_generate_v5$function$
;

-- DROP FUNCTION public.uuid_nil();

CREATE OR REPLACE FUNCTION public.uuid_nil()
 RETURNS uuid
 LANGUAGE c
 IMMUTABLE PARALLEL SAFE STRICT
AS '$libdir/uuid-ossp', $function$uuid_nil$function$
;

-- DROP FUNCTION public.uuid_ns_dns();

CREATE OR REPLACE FUNCTION public.uuid_ns_dns()
 RETURNS uuid
 LANGUAGE c
 IMMUTABLE PARALLEL SAFE STRICT
AS '$libdir/uuid-ossp', $function$uuid_ns_dns$function$
;

-- DROP FUNCTION public.uuid_ns_oid();

CREATE OR REPLACE FUNCTION public.uuid_ns_oid()
 RETURNS uuid
 LANGUAGE c
 IMMUTABLE PARALLEL SAFE STRICT
AS '$libdir/uuid-ossp', $function$uuid_ns_oid$function$
;

-- DROP FUNCTION public.uuid_ns_url();

CREATE OR REPLACE FUNCTION public.uuid_ns_url()
 RETURNS uuid
 LANGUAGE c
 IMMUTABLE PARALLEL SAFE STRICT
AS '$libdir/uuid-ossp', $function$uuid_ns_url$function$
;

-- DROP FUNCTION public.uuid_ns_x500();

CREATE OR REPLACE FUNCTION public.uuid_ns_x500()
 RETURNS uuid
 LANGUAGE c
 IMMUTABLE PARALLEL SAFE STRICT
AS '$libdir/uuid-ossp', $function$uuid_ns_x500$function$
;