import optuna
from joblib import Parallel, delayed
import backtrader as bt

def objective_single_trial(study_name, storage, df):
    """Run one trial"""
    study = optuna.load_study(study_name=study_name, storage=storage)
    trial = study.ask()

    # Your backtrader code here
    data = df.copy()  # or use copy-on-write
    # ... setup cerebro, run backtest ...
    result = cerebro.run()
    value = calculate_metric(result)

    study.tell(trial, value)
    return value

if __name__ == '__main__':
    df = load_data()

    storage = 'sqlite:///backtrader_opt.db'
    study_name = 'strategy_opt'

    study = optuna.create_study(
        direction='maximize',
        storage=storage,
        study_name=study_name,
        load_if_exists=True
    )

    # Run 1000 trials, 6 at a time
    n_trials = 1000
    n_jobs = 6

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(objective_single_trial)(study_name, storage, df)
        for _ in range(n_trials)
    )