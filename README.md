# Description
Модуль предназначен для дизайна и оценки онлайн и оффлайн экспериментов, а также
для дизайна таких экспериментов.

# Repo structure

    📁 TestLab/
    ├─📁 TestLab/                                                           <--- папка с основным модулем
    │ ├─📄 TestLab.py                                                       <--- основной модуль онлайн экспериментов
    │ └─📄 __init__.py
    ├─📁 development/
    │ ├─📄 dev-class-methods-testing.ipynb                                  <--- тестирование методов класса
    │ ├─📄 dev-offline-tests-pipline.ipynb                                  <--- сборка пайплайна тестов
    │ ├─📄 dev-offline-units-exp.ipynb                                      <--- EDA и первичная оценка
    │ ├─📄 dev-testing-etna.ipynb                                           <--- тестирование etna для временных рядов
    │ ├─📄 dev-validation.ipynb                                             <--- пайплайн валидации
    │ ├─📄 src.py                                                           <--- вспомогательные функции
    │ ├─📄 test-data.csv                                                    <--- тестовые данные
    │ ├─📄 test_flow.py                                                     <--- класс с базовыми методами
    │ └─📄 visualization.py                                                 <--- методы визуализации
    ├─📁 playbooks/
    │ └─📄 testing-demo.ipynb                                               <--- демо ноутбук для онлайн экспериментов
    ├─📁 tests/                                                             <--- папка для тестов (тесты не реализованы)
    │ ├─📄 __init__.py
    │ └─📄 test_TestLab.py
    ├─📄 HISTORY.md                                                         <--- история релизов
    ├─📄 LICENSE                                                            <--- лицензия
    ├─📄 MANIFEST.in                                                        <--- манифест
    ├─📄 requirements.txt                                                   <--- требования
    ├─📄 setup.cfg                                                          <--- сетап конфиг
    └─📄 setup.py                                                           <--- сетап
