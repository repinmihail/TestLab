"""Main module."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import scipy.stats as sps
import statsmodels.stats.multitest


from scipy import stats
from IPython.display import display
from scipy.stats import norm
from datetime import datetime, timedelta
from tqdm.notebook import tqdm as tqdm_notebook
from scipy.stats import kurtosis
from scipy.stats import skew
from typing import Tuple, List, Callable, Union
from loguru import logger
from omegaconf import OmegaConf, listconfig
from dataclasses import dataclass


@dataclass
class ExperimentComparisonResults:
    pvalue: float
    effect: float
    ci_length: float
    left_bound: float
    right_bound: float


class ABCore:
    DESCRIPTION_ARGS = pd.DataFrame(
        {
            'stratification_cols': ['поля со стратами', 'List[str] или None'],
            'id_field_name': ['поле с идентификатором (например, subs_id)', 'str'],
            'groups_count': [
                'количество групп в тесте с учетом (и контрольной, и таргетных)', 'int'
            ],
            'control_share': ['доля контрольной группы от размера выборки', 'float'],
            'r': ['отношение самой маленькой группы к самой большой группе (в долях)', 'float'],
            'alpha': ['уровень ошибки I рода', 'float'],
            'beta': ['уровень ошибки II рода', 'float'],
            'sample_size': ['размер выборки для всех групп теста', 'int или None'],
            'mde': ['размер эффекта', 'float или None'],
            'effect_bounds': [
                'границы размеров эффекта в виде массива, \
                например [1.01, 1.02, 1.03, ...]', 'List[float] или None'
            ],
            'sample_size_bounds': [
                'границы размеров выборки в виде массива, например \
                [1000, 5000, 10_000, 50_000]', 'List[int] или None'
            ],
            'metric_dict': [
                'параметр содержит два словаря: target_metric_calc - целевая метрика, \
                help_metric_calc - вспомогательные метрики', 'dict'
            ],
            'target_metric_calc': [
                'словарь, где в качестве ключа - наименование поля с метрикой, \
                а в значении формула вычисления, например, "sum", а для дроби \
                [numerator, denominator]', 'dict'
            ],
            'help_metric_calc': [
                'те же правила + если метрики считаются на одном поле но в разных вариациях, \
                то после имени поля указать постфикс _1, _2 и т.д., например sales_1', 'dict'
            ],
            'n_iter': ['количество итераций в A/A/B тесте', 'int'],
            'weights': ['словарь весов страт', 'dict'],
            'test': [
                'критерий тестирования (simple_ttest, absolute_ttest, welch_ttest)', 'str'
            ],
            'stratification': ['True, если нужна стратификация', 'bool'],
            'stratified_sampling': [
                'True, если в A/A/B тесте нужно использовать стратифицированное\
                сэмплиирование', 'bool'
            ]
        },
        index=['Описание параметра', 'Тип данных']
    ).T
    # параметры для визуализации
    TITLESIZE = 15
    LABELSIZE = 15
    LEGENDSIZE = 12
    XTICKSIZE = 12
    YTICKSIZE = XTICKSIZE

    def __init__(self, design_dict: dict = None) -> None:
        """
        Инициализация базовых атрибутов класса.
        """
        if design_dict:
            self._input_params = OmegaConf.create(design_dict)
            self.target_metric_count = [
                key for key, val in self._input_params.metric_dict.target_metric_calc.items()
            ]
            self.target_share = (
                (1 - self._input_params.control_share) / (self._input_params.groups_count - 1)
            )
            self.r = (
                (1 - self._input_params.control_share) / (self._input_params.groups_count - 1)
                / self._input_params.control_share
            )
            if self._input_params.effect_bounds is None:
                self.effect_bounds = np.linspace(1.01, 1.2, num=10)
            else:
                self.effect_bounds = np.array(self._input_params.effect_bounds)
            if self._input_params.sample_size_bounds is None:
                self.sample_size_bounds = np.arange(1000, 20000, 2000)
            else:
                self.sample_size_bounds = np.array(self._input_params.sample_size_bounds)
        self.groups_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L'
        ]  # Группа A - контрольная
        self.realized_tests = {'simple_ttest', 'absolute_ttest', 'welch_ttest'}
        # для дизайна эксперимента необходимо использовать simple_ttest или welch_ttest
        self.test_types = {
            'simple_ttest': self.simple_ttest,
            'absolute_ttest': self.absolute_ttest,
            'welch_ttest': self.welch_ttest,
        }
        self.stratified_dict_check = dict()
        self.confidence_intervals_methods = {
            'normal': self.get_normal_ci,
            'percentile': self.get_percentile_ci,
            'pivotal': self.get_pivotal_ci,
            'one_tail': self.get_one_tail_ci
        }
        self.confidence_intervals_names = {
            'normal': 'Нормальный',
            'percentile': 'Перцентильный',
            'pivotal': 'Центральный',
            'one_tail': 'Односторонний'
        }

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
        plt.rcParams['axes.labelsize'] = self.LABELSIZE
        plt.rcParams['axes.titlesize'] = self.TITLESIZE
        plt.rcParams['figure.figsize'] = (15, 6)
        plt.rc('xtick', labelsize=self.XTICKSIZE)
        plt.rc('ytick', labelsize=self.YTICKSIZE)
        plt.rc('legend', fontsize=self.LEGENDSIZE)

    @property
    def input_params(self):
        return self._input_params

    def _get_target_metric_calc(self) -> dict:
        """
        Выделяет целевую метрику из заданных параметров.

        return:
            dict - словарь параметров
        """
        return {
            key.split('_')[0]: val for key, val
            in self._input_params.metric_dict['target_metric_calc'].items()
        }

    def _get_help_metrics(self) -> dict:
        """
        Выделяет вспомогательные метрики из заданных параметров.

        args:
            metric_dict - словарь с данными о метриках (собирается из design_dict)
        return:
            dict - словарь параметров
        """
        if self._input_params.metric_dict['help_metric_calc']:
            return {
                key.split('_')[0]: val for key, val
                in self._input_params.metric_dict['help_metric_calc'].items()
            }
        else:
            return dict()

    @logger.catch
    def get_grouped_columns(
        self, data: pd.DataFrame, exception_list: list = None
    ) -> Tuple[list, pd.DataFrame]:
        """
        Собирает те поля датафрейма, по которым будет проводиться агрегация.

        args:
            data - датафрейм с данными для анализа,
            exception_list - лист с полями для исключения (которые не будут участвовать в агрегации)
        return:
            column_for_grouped - лист с полями для группировки,
            data[df_cols] - датафрейм с выбранными полями
        """
        try:
            simple_ratio_metrics = set(self._get_target_metric_calc().keys())
            help_metrics = set(self._get_help_metrics().keys())
            all_metrics = simple_ratio_metrics.union(help_metrics)
        except KeyError as error:
            raise KeyError(f'Проверьте корректность наименования поля: {error}')
        if exception_list:
            column_for_grouped = [
                i for i in data.columns
                if i not in list(all_metrics) + exception_list
            ]
        else:
            column_for_grouped = [
                i for i in data.columns if i not in all_metrics
            ]
        df_cols = column_for_grouped + list(all_metrics)
        return column_for_grouped, data[df_cols]

    @staticmethod
    def print_json(x: dict) -> None:
        """
        Выводит читабельный словарь.

        args:
            x - словарь
        return:
            None
        """
        return print(json.dumps(x, indent=4))

    def _remove_outliers(
        self, data: pd.DataFrame, right_quantile: float = 0.99, left_quantile: float = 0.01
    ) -> pd.DataFrame:
        """
        По значениям 1% и 99% персентилей выбранной метрики обрезает датафрейм.

        args:
            data - датафрейм с данными,
            right_quantile - правый хвост, по умолчанию - 99%,
            left_quantile - левый хвост, по умолчанию - 1%,
        return:
            data - обрезанный датафрейм
        """
        if data[self.target_metric_count[0]].nunique() > 2:
            left_bound = data[self.target_metric_count[0]].quantile(left_quantile)
            right_bound = data[self.target_metric_count[0]].quantile(right_quantile)
            return data[
                (data[self.target_metric_count[0]] > left_bound)
                & (data[self.target_metric_count[0]] < right_bound)
            ]
        else:
            logger.info('Целевая метрика бинарная. Выбросы не обрабатываются.')
            return data

    def delta_method(
        self, data: pd.DataFrame, column_for_grouped: list, is_sample: bool = False
    ) -> dict:
        """
        Рассчитывает дисперсию Дельта-методом. Обычно применяется для ratio-метрик.
        Тестирование метода в playbooks/development/dev-delta-method-validation.ipynb

        args:
            data - датафрейм с данными,
            column_for_grouped - поля для группировки,
            is_sample - если проводится выборочная оценка дисперсии - True, для дизайна - False
        return:
            info_dict - словарь с рассчитанными параметрами
        """

        delta_df = data.groupby(column_for_grouped).agg(
            {self.target_metric_count[0]: ['sum', 'count']}
        )
        n_users = len(delta_df)
        delta_df.columns = ['_'.join(col).strip() for col in delta_df.columns.values]
        array_x = delta_df[f'{self.target_metric_count[0]}_sum'].values
        array_y = delta_df[f'{self.target_metric_count[0]}_count'].values
        mean_x, mean_y = np.mean(array_x), np.mean(array_y)
        var_x, var_y = np.var(array_x), np.var(array_y)
        cov_xy = np.cov(array_x, array_y)[0, 1]
        var_metric = (
            var_x / mean_y ** 2
            - 2 * (mean_x / mean_y ** 3) * cov_xy
            + (mean_x ** 2 / mean_y ** 4) * var_y
        )
        if is_sample:
            var_metric = var_metric / n_users
        info_dict = {}
        info_dict['mean_x, mean_y'] = [mean_x, mean_y]
        info_dict['var_x, var_y'] = [var_x, var_y]
        info_dict['cov_xy'] = cov_xy
        info_dict['n_users'] = n_users
        info_dict['var_metric'] = var_metric
        info_dict['std_metric'] = np.sqrt(var_metric)
        return info_dict

    @staticmethod
    def _linearization_agg(
        data: pd.DataFrame, id_field: str, metric_name: str, total_ids: List = None
    ) -> pd.DataFrame:
        """
        Вспомогательная функция для агрегации данных для линеаризации.

        args:
            data - датафрейм,
            id_field - поле идентификатора для агрегации (например, id пользователя),
            metric_name - поле с метрикой
        return:
            df_lin - датафрейм
        """
        df_agg_metric = (
            data
            .groupby(id_field)[[metric_name]]
            .agg(['sum', 'count'])
            .reset_index()
        )
        df_agg_metric.columns = [id_field, f'{metric_name}_lin_sum', f'{metric_name}_lin_count']
        return df_agg_metric

    def _ratio_metric_calc(
        self, data: pd.DataFrame, key: str, val: Union[str, List], id_field: list,
        postfix: str, kappa: float = None, is_history_data: bool = True,
        linearization_calc: bool = True
    ) -> pd.DataFrame:
        """
        Рассчитывает ratio-метрики в обычном и линеаризованном виде, если это необходимо.

        args:
            data - датафрейм,
            key - метрика,
            val - агрегирующие функции (может быть типа str или list),
            id_field - поле для агрегации,
            postfix - постфикс для наименования метрики,
            kappa - коэффициент kappa,
            is_history_data - True, если на вход подаются исторические данные,
            kappa будет рассчитана на истории,
            linearization_calc - True, если нужна линеаризация
        return:
            agg_df - датафрейм с рассчитанной метрикой
        """
        # расчет ratio-метрики как она есть
        agg_df = data.groupby(id_field).agg({key: val}).reset_index()
        agg_df.columns = [''.join(col).strip() for col in agg_df.columns.values]
        numerator = f'{key}{val[0]}'
        denominator = f'{key}{val[1]}'
        agg_df[f'{key}_ratio_{postfix}'] = agg_df[numerator] / agg_df[denominator]
        ratio_columns = id_field + [f'{key}_ratio_{postfix}']
        if postfix == 'target' and linearization_calc:
            self.ratio_target_metric_as_is = f'{key}_ratio_{postfix}'
        if linearization_calc:
            # расчет линеаризованной метрики
            if kappa is None:
                if is_history_data:
                    df_kappa = self._linearization_agg(data, id_field, key)
                else:
                    control_df = data[data['group'] == 'control']  # наименование контрольной группы
                    df_kappa = self._linearization_agg(control_df, id_field, key)
                kappa = np.sum(df_kappa[f'{key}_lin_sum']) / np.sum(df_kappa[f'{key}_lin_count'])
                logger.info(f'kappa is: {kappa}')
            df_lin = self._linearization_agg(data, id_field, key)
            df_lin[f'{key}_linear_{postfix}'] = (
                df_lin[f'{key}_lin_sum'] - kappa * df_lin[f'{key}_lin_count']
            )
            linear_columns = id_field + [f'{key}_linear_{postfix}']
            agg_df = pd.merge(
                agg_df[ratio_columns],
                df_lin[linear_columns],
                how='left',
                on=id_field
            )
            self.target_metric_name = f'{key}_linear_{postfix}'
            return agg_df, kappa
        else:
            return agg_df[ratio_columns]

    def _user_metric_calc(
        self, data: pd.DataFrame, id_field: list, key: str,
        val: Union[str, List], postfix: str
    ) -> pd.DataFrame:
        """
        Рассчитывает пользовательские метрики.

        args:
            data - датафрейм,
            key - метрика,
            val - агрегирующие функции (может быть типа str или list),
            id_field - поле для агрегации,
            postfix - постфикс для наименования метрики
        return:
            user_metrics - датафрейм с рассчитанной метрикой
        """
        user_metrics = (
            data
            .groupby(id_field)
            .agg({key: val})
            .rename(columns={key: f'{key}_{val}_{postfix}'})
            .reset_index()
        )
        if postfix == 'target':
            self.target_metric_name = f'{key}_{val}_{postfix}'
        return user_metrics

    # решение для метрик - расчет пользовательских, ratio и линеаризованных в одном датафрейме
    def calc_metrics(
        self, data: pd.DataFrame, id_field: list, kappa: float = None,
        is_history_data: bool = True
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Запускает расчет метрик на основе заданных в дизайне формул.

        args:
            data - датафрейм,
            kappa - коэффициент каппа,
            id_field - поле для агрегации,
            is_history_data - True, если на вход подаются исторические данные
        return:
            agg_df - датафрейм с рассчитанными метриками
            kappa_dict - словарь со значениями рассчитанных коэффициентов каппа
        """
        kappa_dict = {}
        for metric_type, metric in self._input_params.metric_dict.items():
            if metric_type == 'target_metric_calc':
                first_metric = True
                for key, val in metric.items():
                    if isinstance(val, list):
                        target_ratio_metrics, kappa = self._ratio_metric_calc(
                            data, key, val, id_field, 'target',
                            kappa=kappa, is_history_data=is_history_data
                        )
                        if first_metric:
                            agg_df = target_ratio_metrics
                            first_metric = False
                        else:
                            agg_df = pd.merge(
                                agg_df,
                                target_ratio_metrics,
                                how='left',
                                on=id_field
                            )
                        kappa_dict[key] = kappa
                        kappa = None
                    else:
                        user_metrics = self._user_metric_calc(
                            data, id_field, key, val, 'target'
                        )
                        if first_metric:
                            agg_df = user_metrics
                            first_metric = False
                        else:
                            agg_df = pd.merge(
                                agg_df,
                                user_metrics,
                                how='left',
                                on=id_field
                            )
            if metric_type == 'help_metric_calc':
                if metric:
                    for key, val in metric.items():
                        key = key.split('_')[0]
                        if isinstance(val, listconfig.ListConfig):
                            help_ratio_metrics = self._ratio_metric_calc(
                                data, key, val, id_field,
                                'help',
                                kappa=kappa,
                                is_history_data=is_history_data,
                                linearization_calc=False
                            )
                            agg_df = pd.merge(
                                agg_df,
                                help_ratio_metrics,
                                how='left',
                                on=id_field
                            )
                        else:
                            help_user_metrics = self._user_metric_calc(
                                data, id_field, key, val, 'help'
                            )
                            agg_df = pd.merge(
                                agg_df,
                                help_user_metrics,
                                how='left',
                                on=id_field
                            )
        return agg_df, kappa_dict

    def data_distribution_plot(
        self, metrics: List[pd.Series], metric_names: List[str],
        verbose: bool = True, main_plot=False
    ) -> None:
        """
        Строит графики распределения для N метрик в одном рисунке.
        Тестирование метода в playbooks/development/dev-plot-improvement.ipynb

        args:
            metrics - лист серий с метриками для отрисовки графиков,
            metric_names - лист с наименованием метрик, которые будут отображаться на графике,
            verbose - True, если нужно отобразить на рисунке параметры распределения,
            main_plot - True, если строится график строится для одной целевой метрики
            (тогда расчет среднего и стандартного отклонения принимается из
            установленных параметров)
        return:
            None - рисунок
        """
        fig, ax = plt.subplots()
        x = 0
        y = -0.12
        step_y = 0
        colors = "bgrcmykw"
        color_index = 0
        for ind, hist in enumerate(metrics):
            if main_plot:
                data_mean = self.params[self.highest_variance_metric]['mean']
                data_std = self.params[self.highest_variance_metric]['std']
            else:
                data_mean = round(np.mean(hist), 4)
                data_std = round(np.std(hist), 4)
            data_median = round(np.median(hist), 4)
            skewness_ = np.round(skew(hist), 4)
            kurtosis_ = np.round(kurtosis(hist), 4)
            lenght = len(hist)
            ax.hist(
                hist,
                bins=300,
                alpha=0.5,
                density=True,
                label=[metric_names[ind]]
            )
            if verbose:
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                textstr = ';  '.join(
                    (
                        f'{metric_names[ind]}',
                        r'$\mu=%.2f$' % (data_mean, ),
                        r'$\mathrm{median}=%.2f$' % (data_median, ),
                        r'$\sigma=%.2f$' % (data_std, ),
                        r'$length=%.0f$' % (lenght, ),
                        r'$skewness=%.2f$' % (skewness_),
                        r'$kurtosis=%.2f$' % (kurtosis_)
                    )
                )
                ax.text(x, y+step_y, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
            step_y -= 0.06
            ax.axvline(
                x=data_mean, ymin=0, ymax=0.90, linestyle='--',
                color=colors[color_index], linewidth=0.7, label=f'mean of {metric_names[ind]}'
            )
            ax.axvline(
                x=data_median, ymin=0, ymax=0.90, linestyle='--', color=colors[color_index+1],
                linewidth=0.7, label=f'median of {metric_names[ind]}'
            )
            color_index += 2
        plt.grid(color='grey', linestyle='--', linewidth=0.2)
        plt.title('Оценка распределения сырых данных', size=16)
        plt.ylabel('Плотность распределения', size=12)
        plt.xlabel('Распределение метрики', size=12)
        plt.legend(loc='upper right')
        plt.show()

    def get_sample_size_complex(self, mu: float, std: float, mde: float) -> Tuple[float, str]:
        """
        Функция для расчета размера выборки при неравных группах.
        Возвращает sample_size для обычной пользовательской метрики, при заданных параметрах теста.
        Тестирование метода в playbooks/development/dev-complex-calc-sample-size-validation.ipynb
        ---> ТАРГЕТНЫЕ ГРУППЫ ДОЛЖНЫ БЫТЬ РАВНЫ <---

        args:
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            mde - мин. детектируемый эффект
        return:
            sample_size - размер всей выборки для теста,
            'complex' - тип метода
        """
        t_alpha = stats.norm.ppf(1 - self._input_params.alpha / 2, loc=0, scale=1)
        t_beta = stats.norm.ppf(1 - self._input_params.beta, loc=0, scale=1)
        sample_ratio_correction = self.r + 2 + 1 / self.r
        comparisons = self._input_params.groups_count - 1
        mu_diff_squared = (mu - mu * mde)**2
        sample_size = (
            sample_ratio_correction * (
                (t_alpha + t_beta)**2) * (std**2)
        ) / (
            mu_diff_squared * (1 - self.target_share * (comparisons - 1))
        )
        return int(np.ceil(sample_size)), 'complex'

    def get_sample_size_standart(self, mu: float, std: float, mde: float) -> Tuple[float, str]:
        """
        Классическая формула расчета размера выборок для двух групп.

        args:
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            mde - мин. детектируемый эффект
        return:
            sample_size - размер выборки для теста,
            'standart' - тип метода
        """
        t_alpha = abs(norm.ppf(1 - self._input_params.alpha / 2, loc=0, scale=1))
        t_beta = norm.ppf(1 - self._input_params.beta, loc=0, scale=1)
        mu_diff_squared = (mu - mu * mde) ** 2
        z_scores_sum_squared = (t_alpha + t_beta) ** 2
        disp_sum = 2 * (std ** 2)
        sample_size = int(
            np.ceil(
                z_scores_sum_squared * disp_sum / mu_diff_squared
            )
        )
        return sample_size, 'standart'

    def get_sample_size_matrix(
        self, sample_size_func: Callable, mu: float, std: float, plot: bool = False
    ) -> pd.DataFrame:
        """
        Строит матрицу значений размера выборки в зависимости от mde.

        args:
            sample_size_func - функция расчета размера выборки,
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            plot - True, если нужно построить график
        return:
            df_res - датафрейм
        """
        res = []
        for eff in self.effect_bounds:
            sample_size, sample_size_method = sample_size_func(mu, std, eff)
            res.append((eff, sample_size, sample_size_method))
        return pd.DataFrame(res, columns=['effects', 'sample_size', 'calc_method'])

    def get_sample_size_plot(self) -> None:
        """
        Функция для построения графика зависимости размера выборки от mde.
        """
        fig, ax = plt.subplots()
        ax.plot(
            (self.sample_size_matrix['effects'].round(4) - 1)*100,
            self.sample_size_matrix['sample_size'],
            'go-', label='sample size', linewidth=2
        )
        plt.grid(color='grey', linestyle='--', linewidth=0.2)
        plt.legend(loc='upper right')
        plt.title('Оценка размера выборки', size=16)
        plt.ylabel('Размер выборки, шт.', size=12)
        plt.xlabel('Эффект, %', size=12)
        plt.show()

    def get_effects_plot(self) -> None:
        """
        Функция для построения графика зависимости размера mde от размера выборки.
        """
        fig, ax = plt.subplots()
        ax.plot(
            self.effects_matrix['sample_size'],
            self.effects_matrix['effect_percent'],
            'go-', label='effect_percent', linewidth=2
        )
        plt.grid(color='grey', linestyle='--', linewidth=0.2)
        plt.legend(loc='upper right')
        plt.title('Оценка размера эффектов', size=16)
        plt.ylabel('Эффект, %', size=12)
        plt.xlabel('Размер выборки, шт.', size=12)
        plt.show()

    def get_MDE(self, mu: float, std: float, sample_size: int) -> Tuple[float, float]:
        """
        Возвращает MDE для обычной пользовательской метрики, при заданных параметрах теста.
        ---> ТАРГЕТНЫЕ ГРУППЫ ДОЛЖНЫ БЫТЬ РАВНЫ <---
        args:
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            sample_size - размер выборки для теста (включает в себя все группы)
        return:
            mde, mde*100/mu - mde в абсолютных и относительных значениях
        """
        t_alpha = stats.norm.ppf(1 - self._input_params.alpha / 2, loc=0, scale=1)
        comparisons = self._input_params.groups_count - 1
        t_beta = stats.norm.ppf(1 - self._input_params.beta, loc=0, scale=1)
        sample_ratio_correction = self.r + 2 + 1 / self.r
        mde = (
            np.sqrt(sample_ratio_correction) * (t_alpha + t_beta) * std /
            np.sqrt(sample_size * (1-self.target_share*(comparisons-1)))
        )
        return mde, mde * 100 / mu

    def get_effects_matrix(
        self, mde_func: Callable, mu: float, std: float, plot: bool = False
    ) -> pd.DataFrame:
        """
        Строит матрицу значений размера эффектов в зависимости от размера выборки.

        args:
            mde_func - функция расчета эффекта,
            sample_size_bounds - границы размера выборки,
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            plot - True, если нужно построить график
        return:
            df_res - датафрейм
        """
        res = []
        for size in self.sample_size_bounds:
            effect_abs, effect_percent = mde_func(mu, std, size)
            res.append((size, effect_abs, effect_percent))
        return pd.DataFrame(res, columns=['sample_size', 'effect_abs', 'effect_percent'])

    @staticmethod
    def get_one_tail_ci(
        mu: float, std: float, sample_size: float, confidence_level: float = 0.95,
        tails: int = 1
    ) -> Tuple[float, float]:
        """
        Вычисляет доверительный интервал.

        args:
            mu - среднее выборки на исторических данных,
            std - стан. отклонение выборки на исторических данных,
            sample_size - размер выборки для теста (включает в себя все группы),
            confidence_level - уровень доверительного интервала,
            tails - двусторонняя или односторонняя проверка
        return
            (left_bound, right_bound): границы доверительного интервала.
        """
        significance_level = 1 - confidence_level
        cum_probability = 1 - (significance_level / tails)
        z_star = norm.ppf(cum_probability)
        left_bound = mu - z_star * std / np.sqrt(sample_size)
        right_bound = mu + z_star * std / np.sqrt(sample_size)
        return left_bound, right_bound

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

    def get_percentile_ci(
        self, bootstrap_stats: Union[np.array, List], pe: float, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Строит перцентильный доверительный интервал.

        args:
            bootstrap_stats - массив значений посчитанной метрики,
            pe - точечная оценка (рассчитывается на исходных данных)

        return:
            left, right - левая и правая границы доверительного интервала
        """
        left, right = np.quantile(bootstrap_stats, [alpha / 2, 1 - alpha / 2])
        return left, right

    def get_pivotal_ci(
        self, bootstrap_stats: Union[np.array, List], pe: float, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Строит центральный доверительный интервал.

        args:
            bootstrap_stats - массив значений посчитанной метрики,
            pe - точечная оценка (рассчитывается на исходных данных)

        return:
            left, right - левая и правая границы доверительного интервала
        """
        left, right = 2 * pe - np.quantile(bootstrap_stats, [1 - alpha / 2,  alpha / 2])
        return left, right

    @logger.catch
    def _stratified_without_group(
        self, data: pd.DataFrame, strata_list: list, group_size: int, weight: float,
        metric_name: str, strat_concat_column: str = 'strat_concat_column',
        group_count: int = 1, verbose: bool = False
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Случайным образом выбирает пользователей в зависимости от веса страты.
        Рассчитывает стратифицированные оценки среднего и дисперсии.

        args:
            data - датафрейм,
            strata_list - страта в формате List[str(strata)],
            group_size - размер выборки,
            weight - веса страт, если известны заранее,
            metric_name - наименование целевой метрики,
            strat_concat_column - поле страт,
            group_count - количество групп,
            verbose - True, если нужен детальный вывод случайных величин
        return:
            random_strata_df - датафрейм страты,
            strat_mean - среднее страты,
            strat_var - дисперсия страты
        """
        np.random.seed(self._input_params.seed)
        strat_df = data[data[strat_concat_column].isin(strata_list)].reset_index(drop=True)
        one_group_size = int(round(group_size * weight)) * group_count
        if one_group_size < 2:
            raise ValueError(
                f'Для страты {strata_list[0]} количество наблюдений в переданном датафрейме '
                f'меньше 2. Необходимо увеличить кол-во наблюдений в страте.'
            )
        if one_group_size > len(strat_df):
            one_group_size = len(strat_df)
        try:
            random_indexes_for_one_group = np.random.choice(
                np.arange(len(strat_df)), one_group_size, False
            )
        except ValueError:
            raise ('Объем выборки превышает объем входных данных.')
        random_strata_df = strat_df.iloc[random_indexes_for_one_group, :]
        strat_mean = random_strata_df.groupby('strat_concat_column')[metric_name].mean() * weight
        strat_var = random_strata_df.groupby('strat_concat_column')[metric_name].var() * weight
        if verbose:
            print(f'Размер датафрейма страты {strata_list}: {len(strat_df)}')
            print(f'one_group_size страты {strata_list}: {one_group_size}')
            print(f'Количество случайных индексов, всего: {len(random_indexes_for_one_group)}')
            print(f'Количество отобранных индексов: {len(random_indexes_for_one_group)}')
            print(f'Количество строк в датафрейме страты: {len(random_strata_df)}')
        return random_strata_df, strat_mean[0], strat_var[0]

    @logger.catch
    def stratify_full_data(
        self, data: pd.DataFrame, group_size: int, metric_name: str, weights: float,
        verbose: bool = False
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Выравнивает общий датафрейм в соответствии с весами, либо переданными в качестве аргумента,
        либо рассчитанными по датафрейму.

        args:
            data - датафрейм с описанием объектов, содержит атрибуты для стратификации,
            group_size - размер одной группы,
            group_count - количество групп (по умолчанию 2),
            weights - словарь весов страт {strat: weight},
                      где strat - страта (strat_concut_columns: str),
                          weight - вес (float)
                      Если None, веса определятся пропорционально доле страт в датафрейме data.
            seed - int, исходное состояние генератора случайных чисел для воспроизводимости
                   результатов. Если None, то состояние генератора не устанавливается.
        return:
            stratified_data - один датафрейм того же формата, что и data,
            количество строк = group_size * group_count
        """
        stratified_df = pd.DataFrame(columns=data.columns)
        stratified_mean = []
        stratified_var = []
        if weights:
            logger.info('Подтягиваю веса из заданного словаря')
            for strata, weight in weights.items():
                strata_list = [strata]
                try:
                    one_strata_df, one_strata_mean, one_strata_var = self._stratified_without_group(
                        data, strata_list, group_size, weight, metric_name
                    )
                    stratified_df = pd.concat([stratified_df, one_strata_df], ignore_index=True)
                    stratified_mean.append(one_strata_mean)
                    stratified_var.append(one_strata_var)
                except UnboundLocalError:
                    raise ('Нарушена индексация. Проверьте границы входимости индексов в датафрейм')
                    return
        else:
            logger.info('Рассчитываю веса страт пропорционально их долям в датафрейме')
            len_data = len(data)
            strat_dict = data.groupby('strat_concat_column').count().iloc[:, 0].to_dict()
            strat_dict_shares = {strata: share/len_data for (strata, share) in strat_dict.items()}
            for strata, weight in strat_dict_shares.items():
                strata_list = [strata]
                try:
                    one_strata_df, one_strata_mean, one_strata_var = self._stratified_without_group(
                        data, strata_list, group_size, weight, metric_name
                    )
                    stratified_df = pd.concat([stratified_df, one_strata_df], ignore_index=True)
                    stratified_mean.append(one_strata_mean)
                    stratified_var.append(one_strata_var)
                except UnboundLocalError:
                    raise ('Нарушена индексация. Проверьте границы входимости индексов в датафрейм')
                    return
        logger.info(
            'Рассчитал среднее и дисперсию: '
            f'{round(np.sum(stratified_mean), 4), round(np.sum(stratified_var), 4)}'
        )
        return stratified_df, np.sum(stratified_mean), np.sum(stratified_var)

    def get_random_group(self, data: pd.DataFrame, group_size: int) -> pd.DataFrame:
        """
        Формирует случайные группы пользователей.

        args:
            data - датафрейм,
            group_size - размер всех групп теста
        return:
            data - датафрейм
        """
        np.random.seed(self._input_params.seed)
        all_groups_df = pd.DataFrame()
        self.groups_weights = self.get_group_weights()
        # выбираем случайные индексы в кол-ве, равном размеру всех групп
        random_indexes = np.random.choice(np.arange(len(data)), group_size, False)
        # для каждой группы определяем кол-во наблюдений и их индексы
        group_indicies_dict = self.get_group_indicies_dict(self.groups_weights, group_size)
        for group_name, current_ind in group_indicies_dict['group_inds'].items():
            if group_name == 'A':
                # из общего массива случайных индексов отбираем все индексы
                # до указанного в значении словаря (current_ind)
                group_inds = random_indexes[:current_ind]
            else:
                # отбираем все индексы от предыдущего до указанного в значении словаря (current_ind)
                group_inds = random_indexes[prev_ind:current_ind]
            prev_ind = current_ind  # запомнили предыдущий
            # отбираем из датафрейма все строки с индексами group_inds
            random_df = data.iloc[group_inds, :]
            random_df['group'] = group_name  # записываем имя группы
            all_groups_df = pd.concat(
                [all_groups_df, random_df],
                ignore_index=True
            )  # записываем в общий дф
        return all_groups_df

    def _get_stratified_with_two_groups(
        self, data: pd.DataFrame, group_size: int, weights: dict
    ) -> pd.DataFrame:
        """
        В соответствии с весами (заданными или рассчитанными) случайным образом
        формирует контрольную и пилотную группы.

        args:
            data - исходный датафрейм,
            group_size - размер группы,
            weights - вес группы
        return:
            data - датафрейм из пилотной и контрольной групп
        """
        np.random.seed(self._input_params.seed)
        pilot = pd.DataFrame(columns=data.columns)
        control = pd.DataFrame(columns=data.columns)
        for strat, weight in weights.items():
            strat_df = data[data['strat_concat_column'].isin([strat])].reset_index(drop=True)
            ab_group_size = int(round(group_size * weight))
            random_indexes_ab = np.random.choice(
                np.arange(len(strat_df)), ab_group_size * 2, False
            )
            a_indexes = random_indexes_ab[:ab_group_size]
            b_indexes = random_indexes_ab[ab_group_size:]
            a_random_strata_df = strat_df.iloc[a_indexes, :]
            b_random_strata_df = strat_df.iloc[b_indexes, :]
            control = pd.concat([control, a_random_strata_df], ignore_index=True)
            pilot = pd.concat([pilot, b_random_strata_df], ignore_index=True)
        control['group'] = 'A'
        pilot['group'] = 'B'
        self.control_strat_dict = (
            control
            .groupby('strat_concat_column')
            .count()
            .iloc[:, 0]
            .to_dict()
        )
        self.pilot_strat_dict = (
            pilot
            .groupby('strat_concat_column')
            .count()
            .iloc[:, 0]
            .to_dict()
        )
        return pd.concat([control, pilot]).reset_index(drop=True)

    def get_group_weights(self):
        """
        Для расчета весов групп.
        """
        groups_weights = [self._input_params.control_share]
        groups_weights.extend((self._input_params.groups_count - 1) * [self.target_share])
        return sorted(groups_weights, reverse=True)

    def get_group_indicies_dict(self, groups_weights: dict, group_size: int) -> dict:
        """
        Формируется словарь, где key - наименование группы из списка groups_names,
        value - индекс группы

        args:
            groups_weights - веса страт,
            group_size - размеры всех групп
        return:
            main_d - словарь с индексами страт для кажэдой группы
        """
        main_d = {}
        group_inds = {}
        for ind, val in enumerate(groups_weights):
            if ind > 0:
                corr_val = prev_val + val * group_size
                prev_val = corr_val
            else:
                prev_val = val * group_size
            group_inds[self.groups_names[ind]] = int(np.ceil(prev_val))
        main_d['group_inds'] = group_inds
        return main_d

    def _stratify_by_multiple_groups(
        self, data: pd.DataFrame, group_size: int, weights: dict, AAB_test: bool = None,
    ) -> pd.DataFrame:
        """
        Функция для стратификации на несколько групп (одна контрольная и несколько пилотных).
        https://habr.com/ru/companies/ozontech/articles/712306/

        args:
            data - исходный датафрейм,
            group_size - размер всех групп теста,
            weights - вес группы,
            AAB_test - True, если используется в A/A/B тесте,
        return:
            data - датафрейм из пилотной и контрольной групп
        """
        np.random.seed(self._input_params.seed)
        all_groups_df = pd.DataFrame()
        self.groups_weights = self.get_group_weights()
        for strat, weight in weights.items():
            # получаем дф с конкретной стратой
            strat_df = data[data['strat_concat_column'].isin([strat])]
            # определяем размер страты для всех групп, размер всех групп * вес страты
            ab_group_size = int(round(group_size * weight))
            # выбираем случайные индексы в кол-ве, равном размеру всех групп
            try:
                random_indexes = np.random.choice(np.arange(len(strat_df)), ab_group_size, False)
            except ValueError:
                raise ValueError(
                    f'Необходимый объем: {ab_group_size}, всего значений в страте: {len(strat_df)}'
                )
            # для каждой группы определяем кол-во наблюдений и их индексы
            group_indicies_dict = self.get_group_indicies_dict(self.groups_weights, ab_group_size)
            for group_name, current_ind in group_indicies_dict['group_inds'].items():
                if group_name == 'A':
                    # из общего массива случайных индексов отбираем все индексы
                    # до указанного в значении словаря (current_ind)
                    group_inds = random_indexes[:current_ind]
                else:
                    # отбираем все индексы от предыдущего до указанного
                    # в значении словаря (current_ind)
                    group_inds = random_indexes[prev_ind:current_ind]
                prev_ind = current_ind  # запомнили предыдущий
                # отбираем из датафрейма все строки с индексами group_inds
                random_strata_df = strat_df.iloc[group_inds, :]
                random_strata_df['group'] = group_name  # записываем имя группы
                all_groups_df = (
                    pd.concat(
                        [all_groups_df, random_strata_df],
                        ignore_index=True
                    )  # записываем в общий дф
                )
        if not AAB_test:
            self.control_strat_dict = (
                all_groups_df[all_groups_df['group'] == 'A']
                .groupby('strat_concat_column')
                .count()
                .iloc[:, 0]
                .to_dict()
            )
            self.pilot_strat_dict = (
                all_groups_df[all_groups_df['group'] == 'B']
                .groupby('strat_concat_column')
                .count()
                .iloc[:, 0]
                .to_dict()
            )
        return all_groups_df

    def get_stratified_groups(
        self, data: pd.DataFrame, group_size: int, weights: dict,
        AAB_test: bool = False
    ) -> pd.DataFrame:
        """
        Запускает стратифицированное сэмплирование.

        args:
            data - исходный датафрейм,
            group_size - размер всех групп теста,
            weights - вес группы,
            AAB_test - True, если используется в A/A/B тесте
        return:
            data - датафрейм из пилотной и контрольной групп
        """
        if weights:
            df = self._stratify_by_multiple_groups(
                data, group_size, weights, AAB_test=AAB_test
            )
            if not AAB_test:
                self.stratified_dict_check['user_weights'] = weights
                self.stratified_dict_check['control_group_weights'] = self.control_strat_dict
                self.stratified_dict_check['pilot_group_weights'] = self.pilot_strat_dict
            return df
        else:
            strat_dict = data.groupby('strat_concat_column').count().iloc[:, 0].to_dict()
            len_data = len(data)
            strat_dict_shares = {
                strata: share/len_data for (strata, share) in strat_dict.items()
            }
            df = self._stratify_by_multiple_groups(
                data, group_size, strat_dict_shares, AAB_test=AAB_test
            )
            if not AAB_test:
                self.stratified_dict_check['calc_weights'] = strat_dict
                self.stratified_dict_check['control_group_weights'] = self.control_strat_dict
                self.stratified_dict_check['pilot_group_weights'] = self.pilot_strat_dict
            return df

    @staticmethod
    def absolute_ttest(
        control: pd.Series, pilot: pd.Series, var_control: float,
        var_pilot: float, alpha: float = 0.05, print_conclusion: bool = False
    ) -> ExperimentComparisonResults:
        """
        Функция абсолютного t-теста. Вычисляет значения pvalue,
        границы доверительного интервала, его длину и эффект.
        Тестирование критерия в playbooks/development/dev-absolute-test.ipynb

        args:
            control - контрольная группа,
            pilot - пилотная группа,
            var_control - оценка дисперсии контрольной группы (может быть стратифицированная),
            var_pilot - оценка дисперсии пилотной группы (может быть стратифицированная),
            alpha - уровень ошибки первого рода,
            print_conclusion - вывод эксперимента
        return:
            pvalue - значение pvalue,
            effect - разница между средними значениями в тестовой и контрольной группах,
            ci_length - длина доверительного интервала,
            left_bound - левая граница доверительного интервала,
            right_bound - правая граница доверительного интервала
        """
        mean_control = np.mean(control)
        mean_pilot = np.mean(pilot)
        var_mean_control = var_control / len(control)
        var_mean_pilot = var_pilot / len(pilot)
        difference_mean = mean_pilot - mean_control
        difference_mean_var = var_mean_control + var_mean_pilot
        difference_distribution = sps.norm(loc=difference_mean, scale=np.sqrt(difference_mean_var))
        left_bound, right_bound = difference_distribution.ppf([alpha/2, 1 - alpha / 2])
        ci_length = (right_bound - left_bound)
        pvalue = 2 * min(difference_distribution.cdf(0), difference_distribution.sf(0))
        effect = difference_mean
        if print_conclusion:
            if pvalue > alpha:
                logger.warning(
                    'Нельзя отвергнуть нулевую гипотезу о равенстве '
                    f'средних при pvalue, равном {pvalue}.'
                )
            else:
                logger.info(
                    'Нулевая гипотеза о равенстве средних отвергается на '
                    f'уровне {alpha} при pvalue, равном {pvalue}.'
                )
        return ExperimentComparisonResults(
            pvalue=pvalue,
            effect=effect,
            ci_length=ci_length,
            left_bound=left_bound,
            right_bound=right_bound
        )

    @staticmethod
    def simple_ttest(
        control: pd.Series, pilot: pd.Series, alpha: float = 0.05,
        print_conclusion: bool = False
    ) -> float:
        """
        Простая функция расчета t-теста из модуля stats.

        args:
            control - контрольная группа,
            pilot - пилотная группа,
            alpha - уровень ошибки первого рода,
            print_conclusion - вывод эксперимента
        return:
            pvalue - значение pvalue
        """
        pvalue = stats.ttest_ind(control, pilot).pvalue
        if print_conclusion:
            if pvalue > alpha:
                logger.warning(
                    'Нельзя отвергнуть нулевую гипотезу о равенстве '
                    f'средних при pvalue, равном {pvalue}.'
                )
            else:
                logger.info(
                    'Нулевая гипотеза о равенстве средних отвергается на '
                    f'уровне {alpha} при pvalue, равном {pvalue}.'
                )
        return pvalue

    @staticmethod
    def tstat_welch_ttest(control: pd.Series, pilot: pd.Series) -> float:
        """
        Функция расчета статистики Welch's t-теста.

        args:
            control - контрольная группа,
            pilot - пилотная группа
        return:
            statistic - значение t-статистики
        """
        xbar_1 = pilot.mean()
        xbar_2 = control.mean()
        var_1 = pilot.var()
        var_2 = control.var()
        n1, n2 = len(pilot), len(control)
        num = xbar_1 - xbar_2
        denom = np.sqrt((var_1/n1) + (var_2/n2))
        return num/denom

    @staticmethod
    def welch_ttest(
        control: pd.Series, pilot: pd.Series, alpha: float = 0.05,
        print_conclusion: bool = False
    ) -> float:
        """
        Простая функция расчета Welch's t-теста из модуля stats.

        args:
            control - контрольная группа,
            pilot - пилотная группа,
            alpha - уровень ошибки первого рода,
            print_conclusion - вывод эксперимента
        return:
            pvalue - значение pvalue
        """
        pvalue = stats.ttest_ind(control, pilot, equal_var=False).pvalue
        if print_conclusion:
            if pvalue > alpha:
                logger.warning(
                    'Нельзя отвергнуть нулевую гипотезу о равенстве '
                    f'средних при pvalue, равном {pvalue}.'
                )
            else:
                logger.info(
                    'Нулевая гипотеза о равенстве средних отвергается на '
                    f'уровне {alpha} при pvalue, равном {pvalue}.'
                )
        return pvalue

    def aa_ab_test_history_calc(
        self, data: pd.DataFrame, sample_size: int, effect_size_abs: float = None,
        test_type: Callable = None
    ) -> Tuple[np.array, np.array]:
        """
        Функция для проведения А/А/B теста на истрических данных для проверки
        корректности подобранного статистического критерия.

        args:
            data - датафрейм,
            sample_size - размер всех групп теста,
            target_metric - наименование целевой метрики,
            effect_size_abs - размер эффекта в абсолютной величине,
            test_type - тип теста,
            n_iter - количество тестов (по умолчанию 10_000)
        return:
            pvalues_aa - массив результатов оценки АА теста,
            pvalues_ab - массив результатов оценки АВ теста
        """
        np.random.seed(self._input_params.seed)
        pvalues_aa = []
        pvalues_ab = []
        for i in tqdm_notebook(range(int(self._input_params.n_iter))):
            # стратифицированное сэмплирование
            if self._input_params.stratified_sampling:
                groups_df = self.get_stratified_groups(
                    data, sample_size, self._input_params.weights, AAB_test=True
                )
                control = groups_df[groups_df['group'] == 'A'][self.target_metric_name]
                test_aa = groups_df[groups_df['group'] == 'B'][self.target_metric_name]
            # нестратифицированное сэмплирование при количестве групп больше 2
            if not self._input_params.stratified_sampling and self._input_params.groups_count > 2:
                groups_df = self.get_random_group(data, sample_size)
                control = groups_df[groups_df['group'] == 'A'][self.target_metric_name]
                test_aa = groups_df[groups_df['group'] == 'B'][self.target_metric_name]
            # нестратифицированное при 2 группах
            if not self._input_params.stratified_sampling and self._input_params.groups_count == 2:
                random_inds = np.random.choice(len(data), sample_size * 2, False)
                random_a = random_inds[:sample_size]
                random_b = random_inds[sample_size:]
                control = data[self.target_metric_name][random_a]
                test_aa = data[self.target_metric_name][random_b]
            if effect_size_abs:
                test_ab = test_aa + effect_size_abs
            pvalue_aa = test_type(control, test_aa)
            pvalue_ab = test_type(control, test_ab)
            pvalues_aa.append(pvalue_aa)
            pvalues_ab.append(pvalue_ab)
        return np.array(pvalues_aa), np.array(pvalues_ab)

    def estimate_ci_bernoulli(self, p: float, n: float) -> Tuple[float, float]:
        """
        Доверительный интервал для Бернуллиевской случайной величины.

        args:
            p - оценка вероятности ошибки (I или II рода),
            n - длина массива
        return:
            Tuple[float, float] - границы доверительного интервала
        """
        t = stats.norm.ppf(1 - self._input_params.alpha / 2, loc=0, scale=1)
        std_n = np.sqrt(p * (1 - p) / n)
        return p - t * std_n, p + t * std_n

    def print_estimated_errors(self, pvalues_aa: np.array, pvalues_ab: np.array) -> pd.DataFrame:
        """
        Оценивает вероятности ошибок.

        args:
            pvalues_aa - массив результатов оценки АА теста,
            pvalues_ab - массив результатов оценки АВ теста
        return:
            df - датафрейм оценок
        """
        self.estimated_first_type_error = np.mean(pvalues_aa < self._input_params.alpha)
        self.estimated_second_type_error = np.mean(pvalues_ab >= self._input_params.alpha)
        self.ci_first = self.estimate_ci_bernoulli(
            self.estimated_first_type_error, len(pvalues_aa)
        )
        self.ci_second = self.estimate_ci_bernoulli(
            self.estimated_second_type_error, len(pvalues_ab)
        )
        self.error_estimation_df = pd.DataFrame(
            {
                'Оценка вероятности ошибки I рода': [
                    f'{self.estimated_first_type_error:0.4f}'
                ],
                'Доверительный интервал ошибки I рода': [
                    f'[{self.ci_first[0]:0.4f}, {self.ci_first[1]:0.4f}]'
                ],
                'Оценка вероятности ошибки II рода': [
                    f'{self.estimated_second_type_error:0.4f}'
                ],
                'Доверительный интервал ошибки II рода': [
                    f'[{self.ci_second[0]:0.4f}, {self.ci_second[1]:0.4f}]'
                ],
            }, index=['Значения']
        ).T
        display(self.error_estimation_df)
        if self.ci_first[0] > 0.05:
            logger.warning(
                'Нельзя отвергнуть нулевую гипотезу о равенстве средних, '
                f'{self.estimated_first_type_error} не входит в доверительный интервал.'
            )
        else:
            logger.info(
                'Нулевая гипотеза о равенстве средних отвергается на уровне '
                f'{self._input_params.alpha} при pvalue, равном {self.estimated_first_type_error}.'
            )
        if self.ci_second[0] > 0.2:
            logger.warning(
                f'Мощность теста недостаточная: {self.estimated_second_type_error} не входит в'
                ' доверительный интервал, вероятность ошибки II рода может быть завышена.'
            )
        return

    def plot_pvalue_distribution(self, pvalues_aa: np.array, pvalues_ab: np.array) -> None:
        """
        Рисует графики распределения p-value.
        Тестирование метода в playbooks/development/dev-plot-improvement.ipynb

        args:
            pvalues_aa - массив результатов оценки АА теста,
            pvalues_ab - массив результатов оценки АВ теста
        return:
            None
        """
        estimated_first_type_error = np.mean(pvalues_aa < self._input_params.alpha)
        estimated_second_type_error = np.mean(pvalues_ab >= self._input_params.alpha)
        y_one = estimated_first_type_error
        y_two = 1 - estimated_second_type_error
        X = np.linspace(0, 1, 1000)
        Y_aa = [np.mean(pvalues_aa < x) for x in X]
        Y_ab = [np.mean(pvalues_ab < x) for x in X]
        plt.plot(X, Y_aa, label='A/A')
        plt.plot(X, Y_ab, label='A/B')
        plt.plot(
            [self._input_params.alpha, self._input_params.alpha],
            [0, 1],
            '-.k',
            alpha=0.8,
            label='Мощность',
            color='g'
        )
        plt.plot([0, self._input_params.alpha], [y_one, y_one], '--k', alpha=0.8)
        plt.plot([0, self._input_params.alpha], [y_two, y_two], '--k', alpha=0.8)
        plt.plot([0, 1], [0, 1], '--k', alpha=0.8, label='Распределение ошибки I рода')
        plt.title('Оценка распределения p-value', size=16)
        plt.xlabel('p-value', size=12)
        plt.legend(fontsize=12)
        plt.grid(color='grey', linestyle='--', linewidth=0.2)
        plt.show()

    def strata_merged(self, data: pd.DataFrame, strats: pd.DataFrame) -> pd.DataFrame:
        """
        Соединяет метрику и страты, создает поле strat_concat_column.

        args:
            data - датафрейм с метрикой,
            strats - датафрейм со стратами,
        return:
            res - датафрейм со стратами
        """
        res = pd.merge(data, strats, on=self._input_params.id_field_name, how='inner')
        res['strat_concat_column'] = (
            res[self._input_params.stratification_cols]
            .astype('str')
            .agg('-'.join, axis=1)
        )
        res['period'] = 'history'
        return res

    def _remove_outs(
        self, data: pd.DataFrame, exception_list: list = None
    ) -> Tuple[list, pd.DataFrame]:
        """
        Обработка выбросов, если целевая метрика одна, и формирует поля для группировки.

        args:
            data - датафрейм с метрикой,
            exception_list - лист для исключения полей (например, дата)
            содержит поля, которые не будут учтены в агрегации
        return:
            column_for_grouped - поля для агрегации,
            remove_outliers_df - датафрейм без выбросов
        """
        self.column_for_grouped, df_ness_cols = self.get_grouped_columns(
            data, exception_list=exception_list
        )
        if len(self.target_metric_count) == 1:
            remove_outliers_df = self._remove_outliers(df_ness_cols)
        else:
            remove_outliers_df = df_ness_cols
        return self.column_for_grouped, remove_outliers_df

    @logger.catch
    def _calc_distribution_params(
        self, remove_outliers_df: pd.DataFrame, strata_df: pd.DataFrame,
        calc_metrics_df: pd.DataFrame
    ) -> dict:
        """
        Рассчитывает параметры распределения целевых(-ой) метрик(-и).

        args:
            remove_outliers_df - датафрейм с сырыми данными для расчета дисперсии
            Дельта-методом для ratio-метрик,
            strata_df - датафрейм со стратами,
            calc_metrics_df - датафрейм, с рассчитанными метриками
        return:
            params_dict - словарь с параметрами
        """
        params_dict = {}
        # делаем предположение, что целевая метрика будет одна
        for metric_name, formula in self._input_params.metric_dict['target_metric_calc'].items():
            one_metric_params = {}
            try:
                if isinstance(int(metric_name.split('_')[-1]), int):
                    corr_metric_name = metric_name[:-2]
            except ValueError:
                corr_metric_name = metric_name
                pass
            if isinstance(formula, list) and not self._input_params.stratification:
                logger.info('Расчет параметров распределения без стратификации для ratio-метрики')
                ratio_metrics_dict = self.delta_method(
                    remove_outliers_df, self.column_for_grouped, corr_metric_name
                )
                MEAN = np.mean(abs(self.calc_metrics_df[self.ratio_target_metric_as_is]))
                VAR = ratio_metrics_dict['var_metric']
                STD = ratio_metrics_dict['std_metric']
                logger.info('Оценки получены')
            if isinstance(formula, str) and self._input_params.stratification:
                logger.info(
                    'Расчет параметров распределения со стратификацией для '
                    'пользовательской метрики. Подтягиваю страты ...')
                self.stratification_df = self.strata_merged(calc_metrics_df, strata_df)
                n_users = self.stratification_df[self._input_params.id_field_name].nunique()
                logger.info('Страты готовы. Рассчитываю стратифицированные оценки ...')
                _, MEAN, VAR = self.stratify_full_data(
                    self.stratification_df, int(np.ceil(n_users / 2)), self.target_metric_name,
                    self._input_params.weights
                )
                STD = np.sqrt(VAR)
                logger.info('Оценки получены')
            if isinstance(formula, str) and not self._input_params.stratification:
                logger.info(
                    'Расчет параметров распределения без стратификации для пользовательской метрики'
                )
                MEAN = np.mean(self.calc_metrics_df[self.target_metric_name])
                VAR = np.var(self.calc_metrics_df[self.target_metric_name])
                STD = np.sqrt(VAR)
                logger.info('Оценки получены')
            try:
                one_metric_params['mean'] = MEAN
                one_metric_params['var'] = VAR
                one_metric_params['std'] = STD
                params_dict[metric_name] = one_metric_params
            except UnboundLocalError:
                raise ('Проверьте корректность выбранных методов расчета параметров распределения')
        return params_dict

    def _set_params(self) -> None:
        """
        Вычисляет метрику с наибольшей дисперсией.
        Устанавливает параметры ее распределения как основные.

        return:
            None
        """
        stds = {key: val['std'] for key, val in self.params.items()}
        highest_variance_key = sorted(stds.items(), key=lambda x: x[1], reverse=True)[0][0]
        self.highest_variance_metric = highest_variance_key
        self.MEAN = self.params[highest_variance_key]['mean']
        self.VAR = self.params[highest_variance_key]['var']
        self.STD = self.params[highest_variance_key]['std']
        return

    @logger.catch
    def get_sample_sizes_and_effects(
        self, data: pd.DataFrame, strata_df: pd.DataFrame = None,
        exception_list: List[str] = None, kappa: float = None
    ) -> pd.DataFrame:
        """
        Функция для расчета размера групп и минимального детектируемого эффекта.

        args:
            data - датафрейм с идентификаторами пользователей и метрикой,
            strata_df - датафрейм с идентификаторами пользователей и стратами,
            exception_list - лист с перечнем полей для исключения из анализа,
            kappa - коэффициент kappa
        return:
            None
        """
        start_time = datetime.now()
        logger.info('Начало работы')
        logger.info('Обрабатываю выбросы в данных ...')

        remove_outliers_df = self._remove_outs(data, exception_list)[1]
        logger.info('Работа с выбросами завершена')
        logger.info('Рассчитываю метрики ...')
        self.calc_metrics_df, _ = self.calc_metrics(
            remove_outliers_df, self._input_params.id_field_name, kappa=None, is_history_data=True)
        logger.info('Метрики рассчитаны')
        logger.info('Рассчитываю параметры распределения ...')
        self.params = self._calc_distribution_params(
            remove_outliers_df, strata_df, self.calc_metrics_df
        )
        self._set_params()
        logger.info('Параметры распределения установлены')
        self.sample_size_matrix = self.get_sample_size_matrix(
                self.get_sample_size_complex, self.MEAN, self.STD, plot=False
        )  # рассчитывается размер выборки для всего теста
        self.effects_matrix = self.get_effects_matrix(
                self.get_MDE, self.MEAN, self.STD, plot=False
        )
        self.data_distribution_plot(
            [self.calc_metrics_df[self.target_metric_name]],
            [self.target_metric_name], main_plot=True
        )  # одна целевая метрика
        display(self.sample_size_matrix)
        display(self.effects_matrix)
        time_elapsed = datetime.now() - start_time
        logger.info('Расчет завершен. Затраченное время (hh:mm:ss.ms) {}'.format(time_elapsed))

    @logger.catch
    def run_AAB_test(self):
        """
        Функция для проведения A/A/B теста на исторических данных.
        """
        start_time = datetime.now()
        logger.info('Начало работы')
        if self._input_params.sample_size is None:
            self._input_params.sample_size = int(self.sample_size_matrix.loc[0, 'sample_size'])
        if self._input_params.mde is None:
            self._input_params.mde = float(self.sample_size_matrix.loc[0, 'effects'])
        effect_size_absolute = (
            self.params[self.highest_variance_metric]['mean'] * (self._input_params.mde - 1)
        )
        logger.info(
            f'Детектируем эффект в {round((self._input_params.mde - 1) * 100, 3)}%'
            f' или {round(effect_size_absolute, 3)} единиц в абсолютном выражении'
        )
        logger.info(f'Размер выборки всех групп теста: {self._input_params.sample_size}')
        logger.info('Запуск A/A/B теста ...')
        if self._input_params.stratification:
            self.pvalues_aa_res, self.pvalues_ab_res = self.aa_ab_test_history_calc(
                self.stratification_df,
                self._input_params.sample_size,
                effect_size_abs=effect_size_absolute,
                test_type=self.test_types[self._input_params.test]
            )
        else:
            self.pvalues_aa_res, self.pvalues_ab_res = self.aa_ab_test_history_calc(
                self.calc_metrics_df,
                self._input_params.sample_size,
                effect_size_abs=effect_size_absolute,
                test_type=self.test_types[self._input_params.test]
            )
        self.print_estimated_errors(self.pvalues_aa_res, self.pvalues_ab_res)
        self.plot_pvalue_distribution(self.pvalues_aa_res, self.pvalues_ab_res)
        time_elapsed = datetime.now() - start_time
        logger.info('Расчет завершен. Затраченное время (hh:mm:ss.ms) {}'.format(time_elapsed))
        return

    @logger.catch
    def get_dataframe_with_groups(self):
        """
        Подбор групп для теста.

        return:
            experiment_groups - датафрейм с рассчитанными группами.
        """
        if self._input_params.stratification:
            self.experiment_groups = self.get_stratified_groups(
                self.stratification_df, self._input_params.sample_size, self._input_params.weights
            )
            self.unit_amount_per_strata_check = {
                key: val for key, val in self.stratified_dict_check[
                    'control_group_weights'
                ].items()
                if val <= self._input_params.groups_count
            }
            if self.unit_amount_per_strata_check:
                logger.warning(
                    'Количество наблюдений в стратах меньше количества групп. '
                    'Страты сохранены в атрибут unit_amount_per_strata_check'
                )
        else:
            self.experiment_groups = self.get_random_group(
                self.calc_metrics_df, self._input_params.sample_size
            )
        return self.experiment_groups

    @logger.catch
    def design_test(
        self, data: pd.DataFrame, strata_df: pd.DataFrame, exception_list: List[str] = None
    ) -> pd.DataFrame:
        """
        Основная функция для дизайна теста.

        args:
            data - датафрейм с идентификаторами пользователей и метрикой,
            strata_df - датафрейм с идентификаторами пользователей и стратами,
            exception_list - лист с перечнем полей для исключения из анализа,
            add_effect - размер добавки (0.01 = 1%)
        return:
            pd.DataFrame - датафрейм с контрольной и пилотной группой
        """
        self.start_time = datetime.now()
        logger.info('Начало работы')
        logger.info('Обрабатываю выбросы в данных ...')
        remove_outliers_df = self._remove_outs(data, exception_list)[1]
        logger.info('Выбросы удалены')
        logger.info('Рассчитываю метрики ...')
        self.calc_metrics_df, _ = self.calc_metrics(
            remove_outliers_df, self._input_params.id_field_name, kappa=None, is_history_data=True
        )
        logger.info('Метрики рассчитаны')
        logger.info('Рассчитываю параметры распределения ...')
        self.params = self._calc_distribution_params(
            remove_outliers_df, strata_df, self.calc_metrics_df
        )
        self._set_params()
        logger.info('Параметры распределения установлены')
        self.sample_size_matrix = self.get_sample_size_matrix(
                self.get_sample_size_complex, self.MEAN, self.STD, plot=False
        )
        self.effects_matrix = self.get_effects_matrix(
                self.get_MDE, self.MEAN, self.STD, plot=False
        )
        self.data_distribution_plot(
            [self.calc_metrics_df[self.target_metric_name]],
            [self.target_metric_name],
            main_plot=True
        )  # одна целевая метрика
        display(self.sample_size_matrix)
        display(self.effects_matrix)
        if self._input_params.mde is None:
            self._input_params.mde = float(self.sample_size_matrix.loc[0, 'effects'])
        if self._input_params.sample_size is None:
            self._input_params.sample_size = int(self.sample_size_matrix.loc[0, 'sample_size'])
        effect_size_absolute = (
            self.params[self.highest_variance_metric]['mean'] * (self._input_params.mde - 1)
        )
        logger.info(
            f'Детектируем эффект в {round((self._input_params.mde - 1) * 100, 3)}%'
            f' или {round(effect_size_absolute, 3)} единиц в абсолютном выражении'
        )
        logger.info(f'Размер выборки всех групп теста: {self._input_params.sample_size}')
        logger.info('Запуск A/A/B теста ...')
        if self._input_params.stratification:
            self.pvalues_aa_res, self.pvalues_ab_res = self.aa_ab_test_history_calc(
                self.stratification_df,
                self._input_params.sample_size,
                effect_size_abs=effect_size_absolute,
                test_type=self.test_types[self._input_params.test]
            )
            logger.info('Подбор контрольной и тестовой групп ...')
            self.experiment_groups = self.get_stratified_groups(
                self.stratification_df,
                self._input_params.sample_size,
                self._input_params.weights
            )
            self.unit_amount_per_strata_check = {
                key: val for key, val in self.stratified_dict_check[
                    'control_group_weights'
                ].items()
                if val <= self._input_params.groups_count
            }
            if self.unit_amount_per_strata_check:
                logger.warning(
                    'Количество наблюдений в стратах меньше количества групп. '
                    'Страты сохранены в объект unit_amount_per_strata_check'
                )
        else:
            self.pvalues_aa_res, self.pvalues_ab_res = self.aa_ab_test_history_calc(
                self.calc_metrics_df,
                self._input_params.sample_size,
                effect_size_abs=effect_size_absolute,
                test_type=self.test_types[self._input_params.test]
            )
            logger.info('Подбор контрольной и тестовой групп ...')
            self.experiment_groups = self.get_random_group(
                self.calc_metrics_df, self._input_params.sample_size
            )
        self.print_estimated_errors(self.pvalues_aa_res, self.pvalues_ab_res)
        self.plot_pvalue_distribution(self.pvalues_aa_res, self.pvalues_ab_res)
        time_elapsed = datetime.now() - self.start_time
        logger.info('Дизайн теста завершен')
        logger.info('Затраченное время (hh:mm:ss.ms) {}'.format(time_elapsed))
        return self.experiment_groups

    @staticmethod
    def plot_pvalue_ecdf(pvalues: Union[List[float], np.array], title: str = None) -> None:
        """
        Визуализация распределения p-value гистограммой + cdf

        args:
            pvalues - массив значений p-value
            title - заголовок рисунка
        return:
            None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        if title:
            plt.suptitle(title)
        sns.histplot(pvalues, ax=ax1, bins=50, stat='density')
        ax1.plot([0, 1], [1, 1], 'k--')
        ax1.set_xlabel('p-value')
        ax1.grid(alpha=0.3)
        sns.ecdfplot(pvalues, ax=ax2)
        ax2.set_ylabel('Probability')
        ax2.set_xlabel('p-value')
        ax2.grid(alpha=0.3)

    def bootstrap_test(
        self, *, test_values: np.ndarray, control_values: np.ndarray, bootstrap_group_length: int,
        metric_func: Callable, ci_type: str, n_iter: int = 10_000, alpha: float = 0.05
    ) -> None:
        """
        Метод оценки стат значимых отличий между группами test_values и control_values
        с помощью случайного сэмплирования.
        Тестирование метода в playbooks/development/dev-bootstrap-test.ipynb

        args:
            test_values - массив значений тестовой группы,
            control_values - массив значений контрольной группы,
            bootstrap_group_length - размер одной подвыборки в сэмплировании,
            metric_func - функция расчета метрики (пользовательская или пакетная),
            ci_type - тип доверительного интервала,
            n_iter - количество итераций,
            alpha - уровень ошибки I рода
        returns:
            None
        """
        np.random.seed(self._input_params.seed)

        def _help_function(func: Callable, group: np.ndarray) -> Callable:
            return func(group)
        difference = []
        for i in tqdm_notebook(range(n_iter)):
            test_group = np.random.choice(test_values, bootstrap_group_length)
            control_group = np.random.choice(control_values, bootstrap_group_length)
            test_metric = _help_function(metric_func, test_group)
            control_metric = _help_function(metric_func, control_group)
            difference.append(test_metric - control_metric)
        self.point_estimation = (
            _help_function(metric_func, test_values) - _help_function(metric_func, control_values)
        )
        self.ci = self.confidence_intervals_methods[ci_type](
            difference, self.point_estimation, alpha
        )
        has_effect = not (self.ci[0] < 0 < self.ci[1])
        logger.info(f'Значение метрики изменилось на: {self.point_estimation:0.5f}')
        logger.info(
            f'{((1 - alpha) * 100)}% доверительный интервал: '
            f'({self.ci[0]:0.5f}, {self.ci[1]:0.5f})'
        )
        logger.info(f'Отличия статистически значимые: {has_effect}')
        sns.kdeplot(difference, label='kde статистики')
        plt.plot([self.point_estimation], [0], 'o', c='k', markersize=6, label='точечная оценка')
        d = 0.02
        plt.plot(self.ci, [-d*2, -d*2], label=f'{self.confidence_intervals_names[ci_type]} ДИ')
        plt.grid(alpha=0.3)
        plt.title('Доверительный интервал')
        plt.legend()
        plt.show()

    @staticmethod
    def _help_to_plot_series(
        series: int, series_name: str, metric_name: str, time_series_name: str, series_index=0
    ) -> None:
        """
        Вспомогательный метод для отрисовки time-series графиков.
        """
        palette = list(sns.palettes.mpl_palette('tab10'))
        xs = series[time_series_name]
        ys = series[metric_name]
        plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

    def plot_time_series(
        self, *, df: pd.DataFrame, metric_name: str, grouped_column: str,
        time_series_name: str = 'date', start_line: str = None, end_line: str = None,
        title: str = False
    ) -> None:
        """
        Основной метод визуализации time-series данных.
        Тестирование метода в playbooks/development/dev-timeseries-vizualization.ipynb

        args:
            df - датафрейм с данными типа time-series,
            metric_name - имя визуализируемой метрики,
            grouped_column - наименование поля с разметкой групп,
            time_series_name - наименование поля с временной шкалой,
            start_line - дата, с которой начинается тест,
            end_line - дата, которой тест заканчивается,
            title - подпись графика
        """
        fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
        df_sorted = df.sort_values(time_series_name, ascending=True)
        for i, (series_name, series) in enumerate(df_sorted.groupby([grouped_column])):
            self._help_to_plot_series(series, series_name, metric_name, time_series_name, i)
            fig.legend(title=grouped_column, bbox_to_anchor=(1, 1), loc='upper left')
        sns.despine(fig=fig, ax=ax)
        plt.xlabel(time_series_name)
        plt.xticks(rotation=90)
        plt.grid(alpha=0.3)
        plt.ylabel(metric_name)
        if title:
            plt.title(title)
        if start_line:
            modified_date = datetime.strptime(start_line, "%Y-%m-%d") + timedelta(days=1)
            plt.axvspan(
                start_line, datetime.strftime(
                    modified_date, "%Y-%m-%d"), color='green', alpha=.1)
        if end_line:
            modified_date = datetime.strptime(end_line, "%Y-%m-%d") + timedelta(days=1)
            plt.axvspan(
                end_line, datetime.strftime(modified_date, "%Y-%m-%d"), color='red', alpha=.1)
        plt.show()


class ABEstimation(ABCore):
    """
    Модуль для оценки и визуализации результатов A/B теста
    """

    GROUP_NAMES = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L'
    ]  # Группа A - контрольная

    def __init__(self, *, df_experiment: pd.DataFrame, df_history: pd.DataFrame = None) -> None:
        """
        Инициализация класса

        args:
            df_history - датафрейм с данными предэкспериментального периода
                        (в зависимости от эксперимента)
            df_experiment - датафрейм с данными экспериментального периода
        """
        super().__init__()
        self.df_history = df_history
        self.df_experiment = df_experiment
        self.test_types = {
            'simple_ttest': self.simple_ttest,
            'absolute_ttest': self.absolute_ttest,
            'welch_ttest': self.welch_ttest,
        }

    def calculate_cuped_metric(
        self, *, target_metric_name: str, id_field_name: str, theta: float = None
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
        if self.df_history is None:
            raise (
                'Для применения CUPED используются исторические или прогнозные данные. '
                'Необходимо задать аргумент df_history.'
            )
        prepilot_period = (
            self.df_history[self.df_history['period'] == 'history']
            .sort_values(id_field_name)
        )
        pilot_period = (
            self.df_experiment[self.df_experiment['period'] == 'pilot']
            .sort_values(id_field_name)
        )
        if theta is None:
            theta = self._calculate_theta(
                y_prepilot=np.array(prepilot_period[target_metric_name]),
                y_pilot=np.array(pilot_period[target_metric_name])
                )
        res = pd.merge(
            prepilot_period,
            pilot_period[[id_field_name, target_metric_name]],
            how='inner',
            on=id_field_name
        )
        cols = list(prepilot_period.columns)
        logger.info(f'Theta is: {theta}', )
        res.columns = cols + [f'{target_metric_name}_prepilot']
        res[f'{target_metric_name}_cuped'] = (
            res[target_metric_name] - theta * res[f'{target_metric_name}_prepilot']
        )
        return res

    @staticmethod
    def _calculate_theta(*, y_prepilot: np.array, y_pilot: np.array) -> float:
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

    @staticmethod
    def multitest_correction(
        *, list_of_pvals: List, alpha: float = 0.05, method: str = 'holm', **kwargs
    ) -> None:
        """
        Корректировка p-value для множественной проверки гипотез.

        args:
            list_of_pvals - массив рассчитанных p-value значений
            alpha - уровень ошибки первого рода
            method - метод поправки, default: 'holm', подробнее по ссылке
                https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
        """
        decision, adj_pvals, sidak_aplha, bonf_alpha = statsmodels.stats.multitest.multipletests(
            pvals=list_of_pvals, alpha=alpha, method=method)
        return dict(
            decision=list(decision),
            adjusted_pvals=[np.round(i, 10) for i in adj_pvals],
            sidak_aplha=sidak_aplha,
            bonf_alpha=bonf_alpha
        )

    def estimation_2_group_by_sampling(
        self, *,
        a_sample: Union[pd.Series, np.array],
        b_sample: Union[pd.Series, np.array],
        sample_group_size: int = 100,
        test_type: str = 'simple_ttest',
        n_iter: int = 10_000,
        seed: int = 42
    ) -> dict:
        """
        Метод оценки статистически значимых отличий между двумя выборками с помощью
        случайного сэмплирования

        args:
            a_samlpe - контрольная группа
            b_sample - тестовая группа
            test_type - тип теста, который будет использоваться при сравнении групп
            n_iter - количество итераций
            sample_group_size - размер сэмплируемых групп,
            seed - int, если фиксируем результаты сэмплирования
            P.S. Сэмплирование - с повторениями.
        return:
            dictionary:
                pvalues - массив значений p-value
                ci_lenghts - массив значений длин интервалов
                left_bound - массив значений левых границ
                right_bound - массив значений правых границ
                fpr - false positive rate
        """
        np.random.seed(seed)
        assert len(a_sample) == len(b_sample), 'Размерность выборок отличается'
        pvalues = []
        ci_lenghts = []
        left_bound = []
        right_bound = []
        effect = []
        fpr = 0
        for _ in tqdm_notebook(range(n_iter)):
            a_sample_idx = np.random.choice(sample_group_size, sample_group_size, replace=True)
            b_sample_idx = np.random.choice(sample_group_size, sample_group_size, replace=True)
            a_random_sample = a_sample[a_sample_idx]
            b_random_sample = b_sample[b_sample_idx]
            if test_type == 'absolute_ttest':
                experiment_result = (
                    self.test_types[test_type](
                        control=a_random_sample,
                        pilot=b_random_sample,
                        var_control=np.var(a_random_sample),
                        var_pilot=np.var(b_random_sample)
                    )
                )
                p_val = experiment_result.pvalue
                pvalues.append(experiment_result.pvalue)
                ci_lenghts.append(experiment_result.ci_length)
                left_bound.append(experiment_result.left_bound)
                right_bound.append(experiment_result.right_bound)
                effect.append(experiment_result.effect)
            else:
                p_val = self.test_types[test_type](a_random_sample, b_random_sample)
                pvalues.append(p_val)
                effect.append(b_random_sample.mean() - a_random_sample.mean())
                ci_lenghts.append([])
                left_bound.append([])
                right_bound.append([])
            if p_val < 0.05:
                fpr += 1
        return dict(
            pvalues=pvalues,
            effect=effect,
            ci_lenghts=ci_lenghts,
            left_bound=left_bound,
            right_bound=right_bound,
            fpr=fpr/n_iter
        )

    def calculate_linear_metric(
        self, *, data: pd.DataFrame, id_field: str, metric_name: str,
        control_group_name: str, kappa: float = None, list_ids: List = None
    ) -> pd.DataFrame:
        """
        args:
            data - датафрейм,
            id_field - поле идентификатора для агрегации (например, id пользователя),
            control_group_name - наименование контрольной группы в датафрейме,
            metric_name - наименование поля метрики,
            kappa - коэффициент kappa,
            list_ids - полный список идентификаторов, для которых нужно посчитать метрику,
                заполняется в случае, если в тесте поучаствовали не все пользователи
        return:
            df_lin - датафрейм с линеаризованной метрикой
        """
        if kappa is None:
            control_df = data[data['group'] == control_group_name]
            df_kappa = self._linearization_agg(control_df, id_field, metric_name)
            kappa = (
                np.sum(df_kappa[f'{metric_name}_lin_sum']) /
                np.sum(df_kappa[f'{metric_name}_lin_count'])
            )
            logger.info(f'kappa is: {kappa}')
        df_lin = self._linearization_agg(data, id_field, metric_name)
        df_lin[f'{metric_name}_linear_target'] = (
            df_lin[f'{metric_name}_lin_sum'] - kappa * df_lin[f'{metric_name}_lin_count']
        )
        if list_ids:
            df_user = pd.DataFrame({id_field: list_ids})
            return df_user.merge(df_lin, on=id_field, how='outer').fillna(0)
        return df_lin

    @staticmethod
    def _stratified_with_groups(
        *, data: pd.DataFrame, weights: float, metric_name: str,
        strat_concat_column: str = 'strat_concat_column'
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Для каждой группы пользователей рассчитывает стратифицированные
        оценки среднего и дисперсии.

        args:
            data - датафрейм,
            weights - веса страт, если известны заранее,
            metric_name - наименование целевой метрики,
            strat_concat_column - поле страт
        return:
            strat_mean - стратифицированное среднее,
            strat_var - стратифицированная дисперсия
        """
        df = data.groupby('strat_concat_column').agg({metric_name: ['mean', 'var']})
        df.columns = ['_'.join(col) for col in df.columns]
        df = df.reset_index()
        df['strata_weight'] = df['strat_concat_column'].map(weights)
        df['strata_product_mean'] = df[f'{metric_name}_mean'] * df['strata_weight']
        df['strata_product_var'] = df[f'{metric_name}_var'] * df['strata_weight']
        strat_mean = df['strata_product_mean'].sum()
        strat_var = df['strata_product_var'].sum()
        return strat_mean, strat_var

    def get_stratification_params(
        self, *, data: pd.DataFrame, strat_columns: List[str], group_list: List[str],
        metric_name: str, weights: dict = None, group_column_name: str = "group"
    ) -> dict:
        """
        Метод для расчета стратифицированных оценок (среднего и дисперсии)
        на этапе оценки эксперимента (постстратификация).
        Тестирование метода в playbooks/development/dev-poststratification.ipynb

        args:
            data - датафрейм с данными экспериментального периода,
            strat_columns - наименования полей, по которым будет произведена стратификация,
            group_column_name - наименование поля с разметкой групп,
            group_list - наименования групп ['A', 'B'...] или ['test', 'control'],
            metric_name - наименование поля с метрикой,
            weights - словарь весов для каждой страты {"strata1": 0.6, "strata2": 0.4}

            Если страты формируются на основе нескольких полей, то наименования страт в словаре
            weights нужно сконкатенировать через "-", например, страты по полу и ОС телефона:
                {
                    "0-apple": 0.2,
                    "1-apple": 0.25,
                    "0-android": 0.3,
                    "1-android": 0.25
                }
        return:
            result_dict:
                stratified_mean - стратифицированное среднее,
                stratified_var - стратифицированная дисперсия
        """
        data["strat_concat_column"] = (
            data[strat_columns]
            .astype("str")
            .agg("-".join, axis=1)
        )
        result_dict = dict()
        if weights:
            for group in group_list:
                group_data = data[data[group_column_name] == group]
                stratified_mean, stratified_var = self._stratified_with_groups(
                    data=group_data,
                    weights=weights,
                    metric_name=metric_name
                )
                result_dict[group] = dict(
                    stratified_mean=stratified_mean,
                    stratified_var=stratified_var
                )
        else:
            # нежелательный сценарий расчета весов на выборке
            weights = data.groupby("strat_concat_column").count().iloc[:, 0].to_dict()
            weights_shares = {
                strata: share / len(data) for (strata, share) in weights.items()
            }
            for group in group_list:
                group_data = data[data[group_column_name] == group]
                stratified_mean, stratified_var = self._stratified_with_groups(
                    data=group_data,
                    weights=weights_shares,
                    metric_name=metric_name
                )
                result_dict[group] = dict(
                    stratified_mean=stratified_mean,
                    stratified_var=stratified_var
                )
        return result_dict
