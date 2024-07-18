import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import functools
import operator
import warnings
import visualization as viz

from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from functools import reduce
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from typing import List, Callable, Tuple, Union
from dtaidistance import dtw
from datetime import timedelta, datetime
warnings.filterwarnings("ignore")


class ABCore:

    def __init__(self, *, params_dict: dict):
        """
        Инициализация базовых атрибутов класса
        """
        self.target_metric = params_dict["target_metric"]
        self.id_field = params_dict["id_field"]
        self.time_series_field = params_dict["time_series_field"]
        self.number_of_neighbors = params_dict["number_of_neighbors"]
        self.test_units = params_dict["test_units"]
        self.alpha = params_dict["alpha"]
        self.cuped_time = params_dict["cuped_time"]
        self.start_of_test = params_dict["start_of_test"]
        self.days_for_knn = params_dict["days_for_knn"]
        self.days_for_validation = params_dict["days_for_validation"]
        self.days_for_test = params_dict["days_for_test"]
        self.n_iter_bootstrap = params_dict["n_iter_bootstrap"]
        
    def create_periods(self):
        self.end_of_test = (
            datetime.strptime(self.start_of_test, "%Y-%m-%d")
            + timedelta(days=self.days_for_test)
        ).strftime("%Y-%m-%d")
        self.start_of_validation = (
            datetime.strptime(self.start_of_test, "%Y-%m-%d")
            + timedelta(days=-self.days_for_validation)
        ).strftime("%Y-%m-%d")
        self.start_of_knn = (
            datetime.strptime(self.start_of_validation, "%Y-%m-%d")
            + timedelta(days=-self.days_for_knn)
        ).strftime("%Y-%m-%d")
        print(f"Подбор групп: с {self.start_of_knn} по {self.start_of_validation}")
        print(f"Валидация: с {self.start_of_validation} по {self.start_of_test}")
        print(f"Тест: с {self.start_of_test} по {self.end_of_test}")
        return

    def get_scaled_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Метод для масштабирования данных. Используется StandardScaler

        Args:
            data (pd.DataFrame): датафрейм с датами, юнитами и метрикой

        Raises:
            KeyError: В случае, если наименование столбца с метрикой указано
            неверно, бросается исключение

        Returns:
            pd.DataFrame: исходный датафрейм + масштабированная метрика
        """
        try:
            scaled_metric = StandardScaler().fit_transform(data[[self.target_metric]])
        except KeyError:
            raise KeyError(f"Frame data does not contain the field with name {self.target_metric}")
        data[f"scaled_{self.target_metric}"] = scaled_metric
        return data

    def get_vectors(self, data: pd.DataFrame) -> Tuple[dict, dict]:
        """
        Преобразует метрику из датафрейма в векторный вид

        Args:
            data (pd.DataFrame): датафрейм c масштабированной метрикой

        Returns:
            Tuple[dict, dict]:
                dict0 - словарь с наименованием юнита в ключе и вектором в значении
                dict1 - словарь с индексом юнита в ключе и наименованием юнита в значении
        """
        data_vec = data.groupby(self.id_field).agg({f"scaled_{self.target_metric}": list}).reset_index()
        data_vec[f"{self.target_metric}_array"] = [np.array(i) for i in data_vec[f"scaled_{self.target_metric}"]]
        keys = data_vec[self.id_field].tolist()
        vals = data_vec[f"{self.target_metric}_array"].tolist()
        return dict(zip(keys, vals)), dict(zip([i for i in range(0, len(keys))], keys))

    def get_k_neighbors_default(self, vectors: dict, id: str, algorithm='auto') -> dict:
        """
        Возвращает k ближайших соседей для одного заданного юнита

        Args:
            id (str): идентификатор юнита
            vectors (dict): словарь с наименованием юнита в ключе и вектором метрики в значении
            algorithm (str, optional): алгорит подбора соседей. Defaults to 'auto'.

        Returns:
            dict: словарь с индексами юнитов в ключах и расстоянием в значении
        """

        def get_knn(vectors):
            vector_arrays = [list(i) for i in vectors.values()]
            return NearestNeighbors(self.number_of_neighbors, algorithm=algorithm).fit(vector_arrays)

        def get_vector(vectors, id):
            return vectors[id].reshape(1, -1)

        def flatten_neighbour_list(distance, ids):
            dist_list, nb_list = distance.tolist(), ids.tolist()
            return dist_list[0], nb_list[0]

        knn = get_knn(vectors)
        vector = get_vector(vectors, id)
        dist, nb_indexes = knn.kneighbors(vector, self.number_of_neighbors, return_distance=True)
        return_dist, return_nb_indexes = flatten_neighbour_list(dist, nb_indexes)
        return dict(zip(return_nb_indexes, return_dist))

    def get_k_neighbors_dtw(self, data_dict: dict, target_id: str) -> dict:
        """
        Находит ближайших соседей по расстоянию DTW для заданного временного ряда

        Args:
            data_dict (dict): словарь, где ключи — идентификаторы наблюдений, значения — вектора временных рядов
            target_id (str): идентификатор временного ряда, для которого ищутся соседи

        Returns:
            nearest_neighbors (dict): словарь, где ключи — идентификаторы ближайших соседей, значения — расстояния DTW
        """
        target_vector = data_dict[target_id]
        distances = {}
        for key, vector in data_dict.items():
            if key != target_id:
                distances[key] = dtw.distance(vector, target_vector)
        sorted_distances = sorted(distances.items(), key=lambda item: item[1])
        nearest_neighbors = dict(sorted_distances[:self.number_of_neighbors])
        return nearest_neighbors

    def get_all_neighbors_eucl(self, knn_vectors: dict, ids_dict: dict) -> dict:
        """
        Возвращает словарь с ближайшими соседями для всех, поданных на вход юнитов

        Args:
            knn_vectors (dict): словарь с наименованием юнита в ключе и вектором в значении
            ids_dict (dict): словарь с индексом юнита в ключе и наименованием юнита в значении

        Returns:
            dict: словарь с наменованием юнита в ключе и списком соседей в значении
        """
        result_ids = {
            i: [
                ids_dict[j] for j in self.get_k_neighbours_default(
                    i, knn_vectors, self.number_of_neighbors + 1
                ) if ids_dict[j] != i
            ]
            for i in self.test_units
        }
        return result_ids

    def get_all_neighbors_dtw(self, knn_vectors: dict) -> dict:
        """
        Возвращает словарь с ближайшими соседями для всех, поданных на вход юнитов

        Args:
            knn_vectors (dict): словарь с наименованием юнита в ключе и вектором в значении

        Returns:
            dict: словарь с наменованием юнита в ключе и списком соседей в значении
        """
        res = {
            i: [
                j for j in self.get_k_neighbors_dtw(knn_vectors, i)
            ] for i in self.test_units
        }
        return res

    def get_test_control_groups(self, neighbors_dict: dict, adj_control: List[str] = None) -> dict:
        """
        Формирует словарь со списками тестовых и контрольных юнитов в значениях словаря

        Args:
            neighbors_dict (dict): словарь с наменованием юнита в ключе и списком соседей в значении
            adj_control (List[str]): лист с юнитами контрольной группы, скорректированной вручную

        Returns:
            dict: итоговый словарь {test_units: [str], control_units: [str]}
        """
        if not adj_control:
            self.control_units = [[j for j in i if j not in self.test_units][0] for i in neighbors_dict.values()]
        else:
            self.control_units = adj_control
        return dict(
            test_units=self.test_units,
            control_units=self.control_units,
        )

    def get_percentile_ci(self, bootstrap_stats: Union[List[float]]):
        """
        Строит перцентильный доверительный интервал

        Args:
            bootstrap_stats (List[float]): бутстрапированная статистика

        Returns:
            Tuple[float, float]: границы доверительного интервала
        """
        left, right = np.quantile(bootstrap_stats, [self.alpha / 2, 1 - self.alpha / 2])
        return left, right

    def get_normal_ci(
            self, bootstrap_stats: Union[np.array, List], pe: float
    ) -> Tuple[float, float]:
        """
        Строит нормальный доверительный интервал.

        Args:
            bootstrap_stats (Union[np.array, List]): массив значений посчитанной метрики
            pe (float): точечная оценка (рассчитывается на исходных данных)

        Returns:
            Tuple[float, float]: границы доверительного интервала
        """
        z = stats.norm.ppf(1 - self.alpha / 2)
        se = np.std(bootstrap_stats)
        left, right = pe - z * se, pe + z * se
        return left, right

    def prepare_data_for_bootstrap(self, data: pd.DataFrame, is_cuped: bool = False) -> dict:
        """
        Готовит массивы данных и размер бутстрап выборки для бутстрапа

        Args:
            data (pd.DataFrame): датафрейм с целевой метрикой

        Returns:
            dict: словарь с массивами и размером бутстрап выборки
        """
        if is_cuped:
            cuped_metric = f"{self.target_metric}_cuped"
            control_values = np.array(data[data[self.id_field].isin(self.control_units)][cuped_metric])
            test_values = np.array(data[data[self.id_field].isin(self.test_units)][cuped_metric])
        else:
            control_values = np.array(data[data[self.id_field].isin(self.control_units)][self.target_metric])
            test_values = np.array(data[data[self.id_field].isin(self.test_units)][self.target_metric])
        assert len(control_values) == len(test_values)
        n = len(control_values)
        return dict(
            control_values=control_values,
            test_values=test_values,
            n=n
        )
        
    def bootstrap(
            self,
            data: pd.DataFrame,
            metric_func: Callable,
            effect: int,
            is_cuped: bool = False,
            directory_path: str = None,
            test_id: str = None
    ) -> dict:
        """
        Бутстрап

        Args:
            data (pd.DataFrame): датафрейм с целевой метрикой
            metric_func (Callable): статистика расчета целевой метрики
            effect (int): искусственный эффект
            directory_path (str): путь для сохранения рисунка
            test_id (str): идентификатор теста

        Returns:
            dict: словарь с рассчитанными параметрами
        """
        def _help_function(func: Callable, group: np.ndarray) -> Callable:
            return func(group)
        
        # Готовим данные
        vals = self.prepare_data_for_bootstrap(data, is_cuped=is_cuped)
        test_values = vals["test_values"]
        control_values = vals["control_values"]
        bootstrap_group_length = vals["n"]
        
        # Бутстрап
        difference_aa = np.zeros(self.n_iter_bootstrap)
        difference_ab = np.zeros(self.n_iter_bootstrap)
        for i in tqdm(range(self.n_iter_bootstrap)):
            random_values_control = np.random.choice(control_values, bootstrap_group_length, True)
            random_values_test = np.random.choice(test_values, bootstrap_group_length, True)
            random_values_test_with_eff = np.random.choice(test_values + effect, bootstrap_group_length, True)

            control_metric = _help_function(metric_func, random_values_control)
            test_metric = _help_function(metric_func, random_values_test)
            test_metric_with_eff = _help_function(metric_func, random_values_test_with_eff)

            difference_aa[i] = test_metric - control_metric
            difference_ab[i] = test_metric_with_eff - control_metric
        # Расчет точечных оценок
        point_estimation_aa = (
                _help_function(metric_func, test_values)
                - _help_function(metric_func, control_values)
        )
        point_estimation_ab = (
                _help_function(metric_func, test_values + effect)
                - _help_function(metric_func, control_values)
        )
        # Считаем p-value
        adj_diffs_aa = difference_aa - point_estimation_aa
        adj_diffs_ab = difference_ab - point_estimation_ab
        false_positive_aa = np.sum(np.abs(adj_diffs_aa) >= np.abs(point_estimation_aa))
        false_positive_ab = np.sum(np.abs(adj_diffs_ab) >= np.abs(point_estimation_ab))
        p_value_aa_boot = false_positive_aa / self.n_iter_bootstrap
        p_value_ab_boot = false_positive_ab / self.n_iter_bootstrap

        # Расчет доверительных интервалов
        ci_aa = self.get_percentile_ci(difference_aa)
        ci_ab = self.get_percentile_ci(difference_ab)
        has_effect_aa = not (ci_aa[0] < 0 < ci_aa[1])
        has_effect_ab = not (ci_ab[0] < 0 < ci_ab[1])
        viz.plot_ci(
            difference_aa, point_estimation_aa, ci_aa,
            p_value_aa_boot, aa_test=True,
            directory_path=directory_path,
            test_id=test_id
        )
        viz.plot_ci(
            difference_ab, point_estimation_ab, ci_ab,
            p_value_ab_boot, aa_test=False,
            directory_path=directory_path,
            test_id=test_id
            )
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
        Вычисляем Theta

        Args:
            y_prepilot (np.array): значения метрики во время пилота
            y_pilot (np.array): значения ковариант (той же самой метрики) на препилоте

        Returns:
            float: значение коэффициента тета
        """
        covariance = np.cov(y_prepilot.astype(float), y_pilot.astype(float))[0, 1]
        variance = np.var(y_prepilot)
        theta = covariance / variance
        return theta

    def calculate_cuped_metric(self, df_history: pd.DataFrame, df_experiment: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет CUPED метрики

        Args:
            df_history (pd.DataFrame): таблица с данными предпилотными данными
            df_experiment (pd.DataFrame): таблица с данными пилота
            theta (float, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: датафрейм с cuped метрикой
        """
        prepilot_period = (
            df_history[df_history['period'] == 'history']
            .sort_values(self.sortmerge_list)
        )
        pilot_period = (
            df_experiment[df_experiment['period'] == 'pilot']
            .sort_values(self.sortmerge_list)
        )
        theta = self._calculate_theta(
            y_prepilot=np.array(prepilot_period[self.target_metric]),
            y_pilot=np.array(pilot_period[self.target_metric])
            )
        res = pd.merge(
            prepilot_period,
            pilot_period,
            how='inner',
            on=self.sortmerge_list,
            suffixes=["_prepilot", "_pilot"]
        )
        print(f'Theta is: {theta}', )
        res[f'{self.target_metric}_cuped'] = (
            res[f"{self.target_metric}_pilot"] - theta * res[f'{self.target_metric}_prepilot']
        )
        return res

    def check_weekday(self, date: str) -> int:
        """
        Возвращает день недели

        Args:
            date (str): дата в формате %Y-%m-%d

        Returns:
            int: день недели
        """
        given_date = datetime.strptime(date, '%Y-%m-%d')
        day_of_week = (given_date.weekday() + 1)
        return day_of_week

    def sort_merge_for_cuped(
        self,
        pre_pilot_df: pd.DataFrame,
        pilot_df: pd.DataFrame,
        all_groups: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Формирование и сортировка датафрейма для cuped'a

        Args:
            pre_pilot_df (pd.DataFrame): данные предпилотного периода
            pilot_df (pd.DataFrame): данные пилотного периода
            all_groups (dict): все юниты теста

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: данные для cuped'a
        """
        # Определяем день недели
        pre_pilot_df["weekday"] = pre_pilot_df[self.time_series_field].apply(lambda x: self.check_weekday(x))
        pilot_df["weekday"] = pilot_df[self.time_series_field].apply(lambda x: self.check_weekday(x))
        # Все юниты в тесте и контроле
        all_units = functools.reduce(operator.iconcat, all_groups.values(), [])
        # Предпилотный период
        dates_for_lin = sorted(list(set(pre_pilot_df[self.time_series_field].values)))[-self.cuped_time:]
        pre_pilot_df = pre_pilot_df[
            pre_pilot_df[self.time_series_field].isin(dates_for_lin) & 
            pre_pilot_df[self.id_field].isin(all_units)
        ]    
        pilot_df_sort = pilot_df.sort_values([self.id_field, "weekday"])
        pre_pilot_df_sort = pre_pilot_df.sort_values([self.id_field, "weekday"])
        pilot_df_sort["row_number"] = [i for i in range(0, len(pilot_df_sort))]
        pre_pilot_df_sort["row_number"] = [i for i in range(0, len(pre_pilot_df_sort))]
        pilot_df_sort["period"] = "pilot"
        pre_pilot_df_sort["period"] = "history"
        cols = [self.time_series_field, self.id_field, self.target_metric, "weekday", "row_number", "period"]
        self.sortmerge_list = [self.id_field, "row_number"]
        return pilot_df_sort[cols], pre_pilot_df_sort[cols]

    @staticmethod
    def multitest_correction(
            *, list_of_pvals: List, alpha: float = 0.05, method: str = 'holm', **kwargs
    ) -> dict:
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