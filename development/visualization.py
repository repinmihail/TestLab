import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt


def plot_ci(difference, point_estimation, ci):
    ax = sns.kdeplot(difference, label='kde статистики', fill=False, color='crimson')
    kdeline = ax.lines[0]
    plt.plot([point_estimation], [0], 'o', c='k', markersize=6, label='точечная оценка')
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    ax.vlines(point_estimation, 0, np.interp(point_estimation, xs, ys), color='crimson', ls=':')
    ax.fill_between(xs, 0, ys, facecolor='crimson', alpha=0.2)
    ax.fill_between(xs, 0, ys, where=(ci[0] <= xs) & (xs <= ci[1]), interpolate=True, facecolor='crimson', alpha=0.2)
    plt.grid(alpha=0.3)
    plt.title('Доверительный интервал')
    plt.legend()
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