import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Tuple
from datetime import datetime,timedelta

TITLESIZE = 15
LABELSIZE = 15
LEGENDSIZE = 12
XTICKSIZE = 12
YTICKSIZE = XTICKSIZE


# the relative size of legend markers vs. original
plt.style.use('bmh')
plt.rcParams['legend.markerscale'] = 1.5
plt.rcParams['legend.handletextpad'] = 0.5
# the vertical space between the legend entries in fraction of fontsize
plt.rcParams['legend.labelspacing'] = 0.4
# border whitespace in fontsize units
plt.rcParams['legend.borderpad'] = 0.5
plt.rcParams['font.size'] = 12
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = LABELSIZE
plt.rcParams['axes.titlesize'] = TITLESIZE
plt.rcParams['figure.figsize'] = (15, 6)
plt.rc('xtick', labelsize=XTICKSIZE)
plt.rc('ytick', labelsize=YTICKSIZE)
plt.rc('legend', fontsize=LEGENDSIZE)


def plot_ci(
    difference: List[float], point_estimation: float,
    ci: Tuple[float, float], p_value: float, aa_test: bool = False,
    directory_path: str = None, test_id: str = None, 
):
    """
    Визуализирует доверительный интервал.
    Рисунок можно сохранить, указав соответствующие парметры

    Args:
        difference (List[float]): разница бутстрап статистики
        point_estimation (float): точечная оценка
        ci (Tuple[float, float]): доверительный интервал
        p_value (float): значение p-value
        aa_test (bool, optional): тип теста. Defaults to False.
        directory_path (str, optional): путь для сохранения рисунка. Defaults to None.
        test_id (str, optional): отличительные параметры имени рисунка для сохранения. Defaults to None.

    Raises:
        TypeError: существование пути
    """
    ax = sns.kdeplot(difference, label='statistics kde', fill=False, color='crimson')
    kdeline = ax.lines[0]
    plt.plot([point_estimation], [0], 'o', c='k', markersize=6, label='point estimation')
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    ax.vlines(point_estimation, 0, np.interp(point_estimation, xs, ys), color='crimson', ls=':')
    ax.fill_between(xs, 0, ys, facecolor='crimson', alpha=0.2)
    ax.fill_between(xs, 0, ys, where=(ci[0] <= xs) & (xs <= ci[1]), interpolate=True, facecolor='crimson', alpha=0.2)

    props = dict(boxstyle='round', ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8), alpha=0.5)
    x = 0.5
    y = -0.23
    if aa_test:
        test_type = "A/A"
        test_path = "AA-tests"
    else:
        test_type = "A/B"
        test_path = "AB-tests"
    has_effect = not (ci[0] < 0 < ci[1])
    textstr = ' \n'.join(
        (
            f'{test_type} test',
            f'',
            f"Point estimation: {round(point_estimation, 2)}",
            f"CI left bound: {round(ci[0], 2)}",
            f"CI right bound: {round(ci[1], 2)}",
            f"Statistically significant differences: {has_effect}",
            f"P-value from bootstrap is: {p_value}",
        )
    )
    ax.text(x, y, textstr, transform=ax.transAxes, fontsize=12,
            ha="center", va="center", bbox=props)
    plt.grid(alpha=0.3)
    plt.title('Confidence Interval')
    plt.legend()
    if directory_path:
        try:
            string_datetime = datetime.today().strftime("%Y%m%d_%H%M%S")
            plt.savefig(
                f"{directory_path}/{test_path}/{test_id}_{string_datetime}.jpeg",
                bbox_inches='tight'
            )
            plt.close()
        except TypeError:
            raise TypeError(f"Убедитесь, что путь {directory_path} существует")
    else:
        plt.show()

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
    title: str = None, save_fig_name: str = None, **kwargs
):
    """
    Основной метод визуализации time-series данных

    Args:
        df (pd.DataFrame): датафрейм с данными типа time-series
        metric_name (str): наименование визуализируемой метрики
        grouped_column (str, optional): наименование поля с разметкой групп. Defaults to None.
        time_series_name (str, optional): наименование поля с временной шкалой. Defaults to 'date'.
        start_line (str, optional): дата, с которой начинается тест. Defaults to None.
        exp_duration_in_days (int, optional): продолжительность эксперимента в днях. Defaults to 14.
        title (str, optional): подпись графика. Defaults to None.
        save_fig_name (str, optional): имя графика, если нужно сохранить. Defaults to None.
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
    if save_fig_name:
        plt.savefig(save_fig_name, **kwargs)
    plt.show()

def plot_pvalue_distribution(pvalues_aa: np.array, pvalues_ab: np.array, alpha=0.05) -> None:
    """
    Рисует графики распределения p-value.
    Тестирование метода в playbooks/development/dev-plot-improvement.ipynb

    args:
        pvalues_aa - массив результатов оценки АА теста,
        pvalues_ab - массив результатов оценки АВ теста
    return:
        None
    """
    estimated_first_type_error = np.mean(pvalues_aa < alpha)
    estimated_second_type_error = np.mean(pvalues_ab >= alpha)
    y_one = estimated_first_type_error
    y_two = 1 - estimated_second_type_error
    X = np.linspace(0, 1, 1000)
    Y_aa = [np.mean(pvalues_aa < x) for x in X]
    Y_ab = [np.mean(pvalues_ab < x) for x in X]
    plt.plot(X, Y_aa, label='A/A')
    plt.plot(X, Y_ab, label='A/B')
    plt.plot(
        [alpha, alpha],
        [0, 1],
        '-.k',
        alpha=0.8,
        label='Мощность',
        color='g'
    )
    plt.plot([0, alpha], [y_one, y_one], '--k', alpha=0.8)
    plt.plot([0, alpha], [y_two, y_two], '--k', alpha=0.8)
    plt.plot([0, 1], [0, 1], '--k', alpha=0.8, label='Распределение ошибки I рода')
    plt.title('Оценка распределения p-value', size=16)
    plt.xlabel('p-value', size=12)
    plt.legend(fontsize=12)
    plt.grid(color='grey', linestyle='--', linewidth=0.2)
    plt.show()