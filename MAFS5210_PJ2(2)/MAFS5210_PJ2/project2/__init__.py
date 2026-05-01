"""Project 2 implementation package."""

__all__ = [
    "run_full_project2_backtest",
    "run_latest_project2_report",
]


def __getattr__(name: str):
    if name in __all__:
        from .backtest import run_full_project2_backtest, run_latest_project2_report

        return {
            "run_full_project2_backtest": run_full_project2_backtest,
            "run_latest_project2_report": run_latest_project2_report,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
