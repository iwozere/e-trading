CREATE TABLE public.job_runs (
	run_id uuid DEFAULT gen_random_uuid() NOT NULL,
	job_type text NOT NULL,
	job_id int8 NULL,
	user_id int8 NULL,
	status text NULL,
	scheduled_for timestamptz NULL,
	enqueued_at timestamptz DEFAULT now() NULL,
	started_at timestamptz NULL,
	finished_at timestamptz NULL,
	job_snapshot jsonb NULL,
	"result" jsonb NULL,
	"error" text NULL,
	CONSTRAINT runs_pkey PRIMARY KEY (run_id)
);
CREATE UNIQUE INDEX ux_runs_job_scheduled_for ON public.job_runs USING btree (job_type, job_id, scheduled_for);


CREATE TABLE public.job_schedules (
	id int8 GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
	user_id int4 NOT NULL,
	"name" varchar(255) NOT NULL,
	job_type varchar(50) NOT NULL,
	"target" varchar(255) NOT NULL,
	task_params jsonb DEFAULT '{}'::jsonb NOT NULL,
	cron varchar(100) NOT NULL,
	enabled bool DEFAULT true NOT NULL,
	next_run_at timestamptz NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	updated_at timestamptz DEFAULT now() NOT NULL
);

CREATE TABLE public.telegram_broadcast_logs (
	id serial4 NOT NULL,
	message text NOT NULL,
	sent_by varchar(255) NOT NULL,
	success_count int4 NULL,
	total_count int4 NULL,
	created_at timestamptz DEFAULT now() NULL,
	CONSTRAINT telegram_broadcast_logs_pkey PRIMARY KEY (id)
);


CREATE TABLE public.telegram_command_audits (
	id serial4 NOT NULL,
	telegram_user_id varchar(255) NOT NULL,
	command varchar(255) NOT NULL,
	full_message text NULL,
	is_registered_user bool NULL,
	user_email varchar(255) NULL,
	success bool NULL,
	error_message text NULL,
	response_time_ms int4 NULL,
	created_at timestamptz DEFAULT now() NULL,
	CONSTRAINT telegram_command_audits_pkey PRIMARY KEY (id)
);
CREATE INDEX ix_telegram_command_audits_command ON public.telegram_command_audits USING btree (command);
CREATE INDEX ix_telegram_command_audits_created ON public.telegram_command_audits USING btree (created_at);
CREATE INDEX ix_telegram_command_audits_success ON public.telegram_command_audits USING btree (success);
CREATE INDEX ix_telegram_command_audits_telegram_user_id ON public.telegram_command_audits USING btree (telegram_user_id);


CREATE TABLE public.telegram_settings (
	"key" varchar(100) NOT NULL,
	value text NULL,
	CONSTRAINT telegram_settings_pkey PRIMARY KEY (key)
);


CREATE TABLE public.usr_users (
	id int4 DEFAULT nextval('users_id_seq'::regclass) NOT NULL,
	email varchar(100) NULL,
	"role" varchar(20) DEFAULT 'trader'::character varying NOT NULL,
	is_active bool DEFAULT true NULL,
	created_at timestamptz DEFAULT now() NULL,
	updated_at timestamp NULL,
	last_login timestamp NULL,
	telegram_user_id varchar(100) NULL,
	CONSTRAINT users_pkey PRIMARY KEY (id)
);
CREATE INDEX ix_users_email ON public.usr_users USING btree (email);
COMMENT ON TABLE public.usr_users IS 'User accounts and authentication';


CREATE TABLE public.webui_performance_snapshots (
	id serial4 NOT NULL,
	strategy_id varchar(100) NOT NULL,
	"timestamp" timestamptz DEFAULT now() NULL,
	pnl jsonb NOT NULL,
	positions jsonb NULL,
	trades_count int4 DEFAULT 0 NULL,
	win_rate jsonb NULL,
	drawdown jsonb NULL,
	metrics jsonb NULL,
	CONSTRAINT webui_performance_snapshots_pkey PRIMARY KEY (id)
);
CREATE INDEX ix_webui_performance_snapshots_strategy_id ON public.webui_performance_snapshots USING btree (strategy_id);


CREATE TABLE public.webui_system_config (
	id serial4 NOT NULL,
	"key" varchar(100) NOT NULL,
	value jsonb NOT NULL,
	description text NULL,
	created_at timestamptz DEFAULT now() NULL,
	updated_at timestamp NULL,
	CONSTRAINT webui_system_config_pkey PRIMARY KEY (id)
);


CREATE TABLE public.telegram_feedbacks (
	id serial4 NOT NULL,
	user_id int4 NOT NULL,
	CONSTRAINT telegram_feedbacks_pkey PRIMARY KEY (id),
	CONSTRAINT telegram_feedbacks_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usr_users(id) ON DELETE CASCADE
);


CREATE TABLE public.trading_bot_instances (
	id varchar(255) NOT NULL,
	"type" varchar(20) NOT NULL,
	status varchar(20) NOT NULL,
	started_at timestamp NULL,
	last_heartbeat timestamp NULL,
	error_count int4 NULL,
	current_balance numeric(20, 8) NULL,
	total_pnl numeric(20, 8) NULL,
	extra_metadata jsonb NULL,
	created_at timestamptz DEFAULT now() NULL,
	updated_at timestamp NULL,
	config jsonb NOT NULL,
	user_id int4 NOT NULL,
	CONSTRAINT trading_bot_instances_pkey PRIMARY KEY (id),
	CONSTRAINT trading_bot_instances_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usr_users(id) ON DELETE CASCADE
);
CREATE INDEX ix_trading_bot_instances_last_heartbeat ON public.trading_bot_instances USING btree (last_heartbeat);
CREATE INDEX ix_trading_bot_instances_status ON public.trading_bot_instances USING btree (status);
CREATE INDEX ix_trading_bot_instances_type ON public.trading_bot_instances USING btree (type);
COMMENT ON TABLE public.trading_bot_instances IS 'Trading bot configuration and status';


CREATE TABLE public.trading_performance_metrics (
	id varchar(36) NOT NULL,
	bot_id varchar(255) NOT NULL,
	trade_type varchar(10) NOT NULL,
	symbol varchar(20) NULL,
	"interval" varchar(10) NULL,
	entry_logic_name varchar(100) NULL,
	exit_logic_name varchar(100) NULL,
	metrics jsonb NOT NULL,
	calculated_at timestamp NULL,
	created_at timestamptz DEFAULT now() NULL,
	CONSTRAINT trading_performance_metrics_pkey PRIMARY KEY (id),
	CONSTRAINT trading_performance_metrics_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.trading_bot_instances(id) ON DELETE CASCADE
);
CREATE INDEX ix_trading_performance_metrics_bot_id ON public.trading_performance_metrics USING btree (bot_id);
CREATE INDEX ix_trading_performance_metrics_bot_id_calculated_at ON public.trading_performance_metrics USING btree (bot_id, calculated_at);
CREATE INDEX ix_trading_performance_metrics_calculated_at ON public.trading_performance_metrics USING btree (calculated_at);
CREATE INDEX ix_trading_performance_metrics_symbol ON public.trading_performance_metrics USING btree (symbol);
COMMENT ON TABLE public.trading_performance_metrics IS 'Performance metrics for trading strategies';


CREATE TABLE public.trading_positions (
	id varchar(36) NOT NULL,
	bot_id varchar(255) NOT NULL,
	trade_type varchar(10) NOT NULL,
	symbol varchar(20) NOT NULL,
	direction varchar(10) NOT NULL,
	opened_at timestamp NULL,
	closed_at timestamp NULL,
	qty_open numeric(20, 8) DEFAULT 0 NOT NULL,
	avg_price numeric(20, 8) NULL,
	realized_pnl numeric(20, 8) DEFAULT 0 NULL,
	status varchar(12) NOT NULL,
	extra_metadata jsonb NULL,
	CONSTRAINT trading_positions_pkey PRIMARY KEY (id),
	CONSTRAINT trading_positions_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.trading_bot_instances(id) ON DELETE CASCADE
);
CREATE INDEX ix_trading_positions_bot_id ON public.trading_positions USING btree (bot_id);
CREATE INDEX ix_trading_positions_bot_id_status ON public.trading_positions USING btree (bot_id, status);
CREATE INDEX ix_trading_positions_symbol ON public.trading_positions USING btree (symbol);
COMMENT ON TABLE public.trading_positions IS 'Open and closed trading positions';


CREATE TABLE public.trading_trades (
	id varchar(36) NOT NULL,
	bot_id varchar(255) NOT NULL,
	trade_type varchar(10) NOT NULL,
	strategy_name varchar(100) NULL,
	entry_logic_name varchar(100) NOT NULL,
	exit_logic_name varchar(100) NOT NULL,
	symbol varchar(20) NOT NULL,
	"interval" varchar(10) NOT NULL,
	entry_time timestamp NULL,
	exit_time timestamp NULL,
	buy_order_created timestamp NULL,
	buy_order_closed timestamp NULL,
	sell_order_created timestamp NULL,
	sell_order_closed timestamp NULL,
	entry_price numeric(20, 8) NULL,
	exit_price numeric(20, 8) NULL,
	entry_value numeric(20, 8) NULL,
	exit_value numeric(20, 8) NULL,
	"size" numeric(20, 8) NULL,
	direction varchar(10) NOT NULL,
	commission numeric(20, 8) NULL,
	gross_pnl numeric(20, 8) NULL,
	net_pnl numeric(20, 8) NULL,
	pnl_percentage numeric(10, 4) NULL,
	exit_reason varchar(100) NULL,
	status varchar(20) NOT NULL,
	extra_metadata jsonb NULL,
	created_at timestamptz DEFAULT now() NULL,
	updated_at timestamp NULL,
	position_id varchar(36) NULL,
	CONSTRAINT trading_trades_pkey PRIMARY KEY (id),
	CONSTRAINT trading_trades_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.trading_bot_instances(id) ON DELETE CASCADE,
	CONSTRAINT trading_trades_position_id_fkey FOREIGN KEY (position_id) REFERENCES public.trading_positions(id) ON DELETE SET NULL
);
CREATE INDEX ix_trading_trades_bot_id ON public.trading_trades USING btree (bot_id);
CREATE INDEX ix_trading_trades_bot_id_entry_time ON public.trading_trades USING btree (bot_id, entry_time);
CREATE INDEX ix_trading_trades_bot_id_status ON public.trading_trades USING btree (bot_id, status);
CREATE INDEX ix_trading_trades_bot_id_symbol ON public.trading_trades USING btree (bot_id, symbol);
CREATE INDEX ix_trading_trades_entry_time ON public.trading_trades USING btree (entry_time);
CREATE INDEX ix_trading_trades_status ON public.trading_trades USING btree (status);
CREATE INDEX ix_trading_trades_strategy_name ON public.trading_trades USING btree (strategy_name);
CREATE INDEX ix_trading_trades_symbol ON public.trading_trades USING btree (symbol);
CREATE INDEX ix_trading_trades_trade_type ON public.trading_trades USING btree (trade_type);
COMMENT ON TABLE public.trading_trades IS 'Individual trade records';


CREATE TABLE public.usr_auth_identities (
	id int4 DEFAULT nextval('auth_identities_id_seq'::regclass) NOT NULL,
	user_id int4 NOT NULL,
	provider varchar(32) NOT NULL,
	external_id varchar(255) NOT NULL,
	metadata jsonb NULL,
	created_at timestamptz DEFAULT now() NULL,
	CONSTRAINT auth_identities_pkey PRIMARY KEY (id),
	CONSTRAINT auth_identities_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usr_users(id) ON DELETE CASCADE
);
CREATE INDEX ix_auth_identities_provider ON public.usr_auth_identities USING btree (provider);
CREATE INDEX ix_auth_identities_user_id ON public.usr_auth_identities USING btree (user_id);


CREATE TABLE public.usr_verification_codes (
	id int4 DEFAULT nextval('verification_codes_id_seq'::regclass) NOT NULL,
	user_id int4 NOT NULL,
	code varchar(32) NOT NULL,
	sent_time int4 NOT NULL,
	provider varchar(20) DEFAULT 'telegram'::character varying NULL,
	created_at timestamptz DEFAULT now() NULL,
	CONSTRAINT verification_codes_pkey PRIMARY KEY (id),
	CONSTRAINT verification_codes_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usr_users(id) ON DELETE CASCADE
);
CREATE INDEX ix_verification_codes_user_id ON public.usr_verification_codes USING btree (user_id);


CREATE TABLE public.webui_audit_logs (
	id serial4 NOT NULL,
	user_id int4 NOT NULL,
	"action" varchar(100) NOT NULL,
	resource_type varchar(50) NULL,
	resource_id varchar(100) NULL,
	details jsonb NULL,
	ip_address varchar(45) NULL,
	user_agent varchar(500) NULL,
	created_at timestamptz DEFAULT now() NULL,
	CONSTRAINT webui_audit_logs_pkey PRIMARY KEY (id),
	CONSTRAINT webui_audit_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.usr_users(id)
);
CREATE INDEX ix_webui_audit_logs_action ON public.webui_audit_logs USING btree (action);
CREATE INDEX ix_webui_audit_logs_user_id ON public.webui_audit_logs USING btree (user_id);


CREATE TABLE public.webui_strategy_templates (
	id serial4 NOT NULL,
	"name" varchar(100) NOT NULL,
	description text NULL,
	template_data jsonb NOT NULL,
	is_public bool DEFAULT false NULL,
	created_by int4 NOT NULL,
	created_at timestamptz DEFAULT now() NULL,
	updated_at timestamp NULL,
	CONSTRAINT webui_strategy_templates_pkey PRIMARY KEY (id),
	CONSTRAINT webui_strategy_templates_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.usr_users(id)
);
CREATE INDEX ix_webui_strategy_templates_created_by ON public.webui_strategy_templates USING btree (created_by);

