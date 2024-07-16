import math
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import functools
import operator
import warnings

from numpy import floating, number, object_, ndarray, dtype
from numpy._typing import _16Bit

import visualization as viz

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from functools import reduce
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import List, Callable, Tuple, Union, Dict, Any
from dtaidistance import dtw
from datetime import timedelta, datetime
from src import plot_time_series



warnings.filterwarnings("ignore")


class ABEstimation:

    def __init__(self, params_dict: dict = None) -> None:
        """
        Инициализация базовых атрибутов класса
        """
        self.target_metric_list = params_dict["target_metric"]
        self.id_field = params_dict["id_field"]
        self.time_series_field = params_dict["time_series_field"]
        self.number_of_neighbors = params_dict["number_of_neighbors"]
        self.test_units = params_dict["test_units"]
        self.alpha = params_dict["alpha"]
        self.beta = params_dict["beta"]
        self.cuped_time = params_dict["cuped_time"]
        self.start_of_test = params_dict["strat_of_test"]

    def _check_data(self, df: pd.DataFrame):
        """
        """
        df_cols = df.columns
        assert (
                len(
                    [i for i in self.target_metric_list if i in df_cols]
                ) == len(self.target_metric_list)
        )

    def get_scaled_data(self, data: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """
        Метод для масштабирования данных. Используется StandardScaler

        Args:
            data (pd.DataFrame): датафрейм с датами, юнитами и метрикой
            metric_name (str): наменование столбца с метрикой

        Raises:
            KeyError: В случае, если наименование столбца с метрикой указано
            неверно, бросается исключение

        Returns:
            pd.DataFrame: исходный датафрейм + масштабированная метрика
        """
        try:
            scaled_metric = StandardScaler().fit_transform(data[[metric_name]])
        except KeyError:
            raise KeyError(f"Frame data does not contain the field with name {metric_name}")
        data[f"scaled_{metric_name}"] = scaled_metric
        return data

    def get_vectors(
            self, data: pd.DataFrame, metric_name: str, id_field_name: str
    ) -> Tuple[dict, dict]:
        """
        Преобразует метрику из датафрейма в векторный вид

        Args:
            data (pd.DataFrame): датафрейм c масштабированной метрикой
            metric_name (str): наименование столбца с масштабированной метрикой
            id_field_name (str): наименование столбца с юнитом

        Returns:
            Tuple[dict, dict]:
                dict0 - словарь с наименованием юнита в ключе и вектором в значении
                dict1 - словарь с индексом юнита в ключе и наименованием юнита в значении
        """
        data_vec = data.groupby(id_field_name).agg({f"scaled_{metric_name}": list}).reset_index()
        data_vec[f"{metric_name}_array"] = [np.array(i) for i in data_vec[f"scaled_{metric_name}"]]
        keys = data_vec[id_field_name].tolist()
        vals = data_vec[f"{metric_name}_array"].tolist()
        return dict(zip(keys, vals)), dict(zip([i for i in range(0, len(keys))], keys))

    def get_k_neighbours_default(
            self, id: str, vectors: dict, number_of_neighbours: int, algorithm='auto'
    ) -> dict:
        """
        Возвращает k ближайших соседей для одного заданного юнита

        Args:
            id (str): идентификатор юнита
            vectors (dict): словарь с наименованием юнита в ключе и вектором метрики в значении
            number_of_neighbours (int): количество ближайших соседей для поиска
            algorithm (str, optional): алгорит подбора соседей. Defaults to 'auto'.

        Returns:
            dict: словарь с индексами юнитов в ключах и расстоянием в значении
        """

        def get_knn(vectors):
            vector_arrays = [list(i) for i in vectors.values()]
            return NearestNeighbors(number_of_neighbours, algorithm=algorithm).fit(vector_arrays)

        def get_vector(vectors, id):
            return vectors[id].reshape(1, -1)

        def flatten_neighbour_list(distance, ids):
            dist_list, nb_list = distance.tolist(), ids.tolist()
            return dist_list[0], nb_list[0]

        knn = get_knn(vectors)
        vector = get_vector(vectors, id)
        dist, nb_indexes = knn.kneighbors(vector, number_of_neighbours, return_distance=True)
        return_dist, return_nb_indexes = flatten_neighbour_list(dist, nb_indexes)
        return dict(zip(return_nb_indexes, return_dist))

    def get_k_neighbours_dtw(
            self, data_dict: dict, target_id: str, n_neighbors=1
    ):
        """
        Находит ближайших соседей по расстоянию DTW для заданного временного ряда.

        :param data_dict: словарь, где ключи — идентификаторы наблюдений, значения — вектора временных рядов.
        :param target_id: идентификатор временного ряда, для которого ищутся соседи.
        :param n_neighbors: количество ближайших соседей, которые нужно вернуть.
        :return: словарь, где ключи — идентификаторы ближайших соседей, значения — расстояния DTW.
        """
        target_vector = data_dict[target_id]
        distances = {}
        for key, vector in data_dict.items():
            if key != target_id:
                distances[key] = dtw.distance(vector, target_vector)
        sorted_distances = sorted(distances.items(), key=lambda item: item[1])
        nearest_neighbors = dict(sorted_distances[:n_neighbors])
        return nearest_neighbors

    def get_all_neighbours_eucl(
            self, knn_vectors: dict, ids_dict: dict, test_units: List[str],
            number_of_neighbours: int
    ) -> dict:
        """
        Возвращает словарь с ближайшими соседями для всех, поданных на вход юнитов

        Args:
            knn_vectors (dict): словарь с наименованием юнита в ключе и вектором в значении
            ids_dict (dict): словарь с индексом юнита в ключе и наименованием юнита в значении
            test_units (List[str]): список юнитов из тестовой группы
            number_of_neighbours (int): количество ближайших соседей

        Returns:
            dict: словарь с наменованием юнита в ключе и списком соседей в значении
        """
        result_ids = {
            i: [
                ids_dict[j] for j in self.get_k_neighbours_default(
                    i, knn_vectors, number_of_neighbours + 1
                ) if ids_dict[j] != i
            ]
            for i in test_units
        }
        return result_ids

    def get_all_neighbours_dtw(
            self, knn_vectors: dict, number_of_neighbours: int, test_units: List[str]
    ) -> dict:
        """
        Возвращает словарь с ближайшими соседями для всех, поданных на вход юнитов

        Args:
            knn_vectors (dict): словарь с наименованием юнита в ключе и вектором в значении
            number_of_neighbours (int): количество ближайших соседей
            test_units (List[str]): список юнитов из тестовой группы
        """
        res = {
            i: [j for j in self.get_k_neighbours_dtw(knn_vectors, i, number_of_neighbours)] for i in test_units
        }
        return res

    def get_test_control_val_groups(self, neighbours_dict: dict) -> dict:
        """
        Формирует словарь со списками тестовых и контрольных юнитов в значениях словаря

        Args:
            neighbours_dict (dict): словарь с наменованием юнита в ключе и списком соседей в значении

        Returns:
            dict: итоговый словарь {test_units: [str], control_units: [str]}
        """
        test_units = list(neighbours_dict.keys())
        control_units = [[j for j in i if j not in test_units][0] for i in neighbours_dict.values()]
        return dict(
            test_units=test_units,
            control_units=control_units,
        )

    def get_percentile_ci(self, bootstrap_stats: Union[List[float]], alpha: float = 0.05):
        """Строит перцентильный доверительный интервал."""
        left, right = np.quantile(bootstrap_stats, [alpha / 2, 1 - alpha / 2])
        return left, right

    def get_normal_ci(
            self, bootstrap_stats: Union[np.array, List], pe: float, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Строит нормальный доверительный интервал.

        args:
            bootstrap_stats - массив значений посчитанной метрики,
            pe - точечная оценка (рассчитывается на исходных данных)

        return:
            left, right - левая и правая границы доверительного интервала
        """
        z = stats.norm.ppf(1 - alpha / 2)
        se = np.std(bootstrap_stats)
        left, right = pe - z * se, pe + z * se
        return left, right

    def bootstrap(
            self,
            control_group: np.array,
            test_group: np.array,
            metric_func: Callable,
            bootstrap_group_length: int,
            effect: int,
            alpha: float = 0.05,
            n_iter: int = 10_000,
            verbose: bool = False
    ):
        def _help_function(func: Callable, group: np.ndarray) -> Callable:
            return func(group)
        difference_aa = np.zeros(n_iter)
        difference_ab = np.zeros(n_iter)
        # p_value_aa_res_tt = np.zeros(n_iter)
        # p_value_ab_res_tt = np.zeros(n_iter)
        for i in range(n_iter):
            random_values_control = np.random.choice(control_group, bootstrap_group_length, True)
            random_values_test = np.random.choice(test_group, bootstrap_group_length, True)
            random_values_test_with_eff = np.random.choice(test_group + effect, bootstrap_group_length, True)

            control_metric = _help_function(metric_func, random_values_control)
            test_metric = _help_function(metric_func, random_values_test)
            test_metric_with_eff = _help_function(metric_func, random_values_test_with_eff)

            difference_aa[i] = test_metric - control_metric
            difference_ab[i] = test_metric_with_eff - control_metric
        # Расчет точечных оценок
        point_estimation_aa = (
                _help_function(metric_func, test_group) - _help_function(metric_func, control_group)
        )
        point_estimation_ab = (
                _help_function(metric_func, test_group + effect) - _help_function(metric_func, control_group)
        )
        # Считаем p-value
        adj_diffs_aa = difference_aa - point_estimation_aa
        adj_diffs_ab = difference_ab - point_estimation_ab
        false_positive_aa = np.sum(np.abs(adj_diffs_aa) >= np.abs(point_estimation_aa))
        false_positive_ab = np.sum(np.abs(adj_diffs_ab) >= np.abs(point_estimation_ab))
        p_value_aa_boot = false_positive_aa / n_iter
        p_value_ab_boot = false_positive_ab / n_iter

        # Расчет доверительных интервалов
        ci_aa = self.get_percentile_ci(difference_aa)
        ci_ab = self.get_percentile_ci(difference_ab)
        has_effect_aa = not (ci_aa[0] < 0 < ci_aa[1])
        has_effect_ab = not (ci_ab[0] < 0 < ci_ab[1])
        if verbose:
            print("A/A тест")
            print(f'Значение метрики изменилось на: {point_estimation_aa:0.5f}')
            print(
                f'{((1 - alpha) * 100)}% доверительный интервал: '
                f'({ci_aa[0]:0.5f}, {ci_aa[1]:0.5f})'
            )
            print(f'Отличия статистически значимые: {has_effect_aa}')
            print(f"p-value from bootstrap is: {p_value_aa_boot}")
            viz.plot_ci(difference_aa, point_estimation_aa, ci_aa)
            print("------------------")
            print("A/B тест")
            print(f'Значение метрики изменилось на: {point_estimation_ab:0.5f}')
            print(
                f'{((1 - alpha) * 100)}% доверительный интервал: '
                f'({ci_ab[0]:0.5f}, {ci_ab[1]:0.5f})'
            )
            print(f'Отличия статистически значимые: {has_effect_ab}')
            print(f"p-value from bootstrap is: {p_value_ab_boot}")
            viz.plot_ci(difference_ab, point_estimation_ab, ci_ab)
        else:
            return dict(
                aa_test=has_effect_aa,
                ab_test=has_effect_ab,
                pe_aa=point_estimation_aa,
                pe_ab=point_estimation_ab,
                ci_aa=ci_aa,
                ci_ab=ci_ab,
                p_value_aa_boot=p_value_aa_boot,
                p_value_ab_boot=p_value_ab_boot
            )

    def _calculate_theta(self, *, y_prepilot: np.array, y_pilot: np.array) -> float:
        """
        Вычисляем Theta.

        args:
            y_pilot - значения метрики во время пилота,
            y_prepilot - значения ковариант (той же самой метрики) на препилоте
        return:
            theta - значение коэффициента тета
        """
        covariance = np.cov(y_prepilot.astype(float), y_pilot.astype(float))[0, 1]
        variance = np.var(y_prepilot)
        theta = covariance / variance
        return theta

    def calculate_cuped_metric(
            self, df_history, df_experiment, target_metric_name: str, id_field_name: str, theta: float = None
    ) -> pd.DataFrame:
        """
        Вычисляет коварианту и преобразованную метрику cuped.

        args:
            df - pd.DataFrame, датафрейм с данными по пользователям,
            метрикам (нормализованной ключевой метрикой) и стратами с разметкой:
                1) на контроль и пилот (A/B/C..., где A-контроль) - столбец group,
                2) пред-экспериментальный и экспериментальный периоды
                (pilot/prepilot) - столбец period,
        return:
            res - датафрейм
        """
        if df_history is None:
            raise (
                'Для применения CUPED используются исторические или прогнозные данные. '
                'Необходимо задать аргумент df_history.'
            )
        prepilot_period = (
            df_history[df_history['period'] == 'history']
            .sort_values(id_field_name)
        )
        pilot_period = (
            df_experiment[df_experiment['period'] == 'pilot']
            .sort_values(id_field_name)
        )
        if theta is None:
            theta = self._calculate_theta(
                y_prepilot=np.array(prepilot_period[target_metric_name]),
                y_pilot=np.array(pilot_period[target_metric_name])
            )
        res = pd.merge(
            prepilot_period,
            pilot_period,
            how='inner',
            on=id_field_name,
            suffixes=("_prepilot", "_pilot")
        )
        # cols = list(prepilot_period.columns)
        print(f'Theta is: {theta}', )
        res[f'{target_metric_name}_cuped'] = (
                res[f"{target_metric_name}_pilot"] - theta * res[f'{target_metric_name}_prepilot']
        )
        return res

    @staticmethod
    def check_weekday_or_weekend(date):
        given_date = datetime.strptime(date, '%Y-%m-%d')
        day_of_week = (given_date.weekday() + 1)
        return day_of_week

    def sort_merge_for_cuped(
            self, pre_pilot_df: pd.DataFrame, pilot_df: pd.DataFrame, id_field: str, cuped_time: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        """
        pre_pilot_df["weekday"] = pre_pilot_df[self.time_series_field].apply(
            lambda x: self.check_weekday_or_weekend(x))
        pilot_df["weekday"] = pilot_df[self.time_series_field].apply(
            lambda x: self.check_weekday_or_weekend(x))
        dates_for_lin = sorted(list(set(pre_pilot_df[self.time_series_field].values)))[-cuped_time:]
        print(len(dates_for_lin))
        pre_pilot_df = pre_pilot_df[
            pre_pilot_df[self.time_series_field].isin(dates_for_lin) &
            pre_pilot_df[self.id_field].isin(set(pilot_df[self.id_field].values))
            ]
        print(pilot_df.shape, pre_pilot_df.shape)

        pilot_df_sort = pilot_df.sort_values([self.id_field, "weekday"])
        pre_pilot_df_sort = pre_pilot_df.sort_values([self.id_field, "weekday"])

        pilot_df_sort["row_number"] = [i for i in range(0, len(pilot_df_sort))]
        pre_pilot_df_sort["row_number"] = [i for i in range(0, len(pre_pilot_df_sort))]

        pilot_df_sort["period"] = "pilot"
        pre_pilot_df_sort["period"] = "history"
        return pilot_df_sort, pre_pilot_df_sort

    @staticmethod
    def multitest_correction(
            *, list_of_pvals: List, alpha: float = 0.05, method: str = 'holm', **kwargs
    ) -> dict[str, list[Any] | list[
        floating[_16Bit] | number[Any] | object_ | ndarray[Any, dtype[floating[_16Bit]]] | ndarray[
            Any, dtype[Any]] | Any] | Any]:
        """
        Корректировка p-value для множественной проверки гипотез.

        args:
            list_of_pvals - массив рассчитанных p-value значений
            alpha - уровень ошибки первого рода
            method - метод поправки, default: 'holm', подробнее по ссылке
                https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
        """
        decision, adj_pvals, sidak_aplha, bonf_alpha = stats.multitest.multipletests(
            pvals=list_of_pvals, alpha=alpha, method=method)
        return dict(
            decision=list(decision),
            adjusted_pvals=[np.round(i, 10) for i in adj_pvals],
            sidak_aplha=sidak_aplha,
            bonf_alpha=bonf_alpha
        )