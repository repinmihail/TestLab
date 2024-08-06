import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seedir as sd
import emoji


from datetime import datetime, timedelta


def _help_to_plot_series(
    series: int, series_name: str, metric_name: str, time_series_name: str, series_index = 0,
    start_line: str = None
):
    """
    Вспомогательный метод для отрисовки time-series графиков.
    """
    palette = list(sns.palettes.mpl_palette('tab10'))
    xs = series[time_series_name]
    ys = series[metric_name]
    plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

def plot_time_series(
    *, df: pd.DataFrame, metric_name: str, grouped_column: str = None,
    time_series_name: str = 'date', start_line: str = None, exp_duration_in_days: int = 14,
    title: str = False, save_fig: bool = False, fig_name: str = None, **kwargs
):
    """
    Основной метод визуализации time-series данных.

    args:
        df - датафрейм с данными типа time-series,
        metric_name - имя визуализируемой метрики,
        grouped_column - наименование поля с разметкой групп,
        time_series_name - наименование поля с временной шкалой,
        start_line - дата, с которой начинается тест,
        exp_duration_in_days - продолжительность эксперимента в днях,
        title - подпись графика
    """
    fig, ax = plt.subplots(figsize=(10, 5.2))
    df_sorted = df.sort_values(time_series_name, ascending=True)
    if not grouped_column:
        grouped_column = "metrics"
        df_sorted[grouped_column] = metric_name
    for i, (series_name, series) in enumerate(df_sorted.groupby([grouped_column])):
        _help_to_plot_series(series, series_name, metric_name, time_series_name, i)
        ax.legend(title=grouped_column, bbox_to_anchor=(1, 1), loc='upper left')
    sns.despine(fig=fig, ax=ax)
    plt.xlabel(time_series_name)
    plt.xticks(rotation=90)
    plt.grid(alpha=0.3)
    plt.ylabel(metric_name)
    if title:
        plt.title(title)
    if start_line:
        modified_date = datetime.strptime(start_line, "%Y-%m-%d") + timedelta(days=exp_duration_in_days)
        plt.axvspan(
            start_line, datetime.strftime(modified_date, "%Y-%m-%d"),
            color='green', alpha=.1, label="Experiment period"
        )
        ax.legend()
    if save_fig:
        plt.savefig(fig_name, **kwargs)
    plt.show()

def get_directory_tree(path: str):
    """
    Возвращает структуру директории

    Args:
        path (str): путь директории,
        kwargs (dict): аргументы seedir. Default = None
    """
    sd.seedir(
        path,
        first='folders',
        style='emoji',
        exclude_folders=["__pycache__", ".git"]
    )