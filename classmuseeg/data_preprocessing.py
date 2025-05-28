import os
import pandas as pd
import numpy as np
from scipy import signal
import re
from itertools import combinations

def load_data(data_dir, subjects, stims, trials, channels):
    """
    Загружаются данные из CSV-файлов для указанных испытуемых, стимулов, проб и каналов,
    формируются ключи в формате "Sub1_Stim1_Trial1" и сохраняются соответствующие записи в словаре.
    
    Эта функция обрабатывает файлы данных для каждого испытуемого, стимула и пробы, 
    выбирает только указанные каналы и сохраняет данные в словарь с уникальными ключами.
    
    :param data_dir: Путь к директории с CSV-файлами.
    :param subjects: Список номеров испытуемых.
    :param stims: Список стимулов, которые нужно обработать.
    :param trials: Список номеров проб, которые нужно обработать.
    :param channels: Список каналов, которые необходимо сохранить.
    :return: Словарь, где ключ — это идентификатор в формате "Sub1_Stim1_Trial1",
             а значение — DataFrame с данными по выбранным каналам.
    """
    data_dict = {}  # Словарь для хранения данных с ключами формата "SubX_StimX_TrialX"

    # Итерация по всем комбинациям испытуемых, стимулов и проб
    for sub_num in subjects:  # Обрабатывается каждый испытуемый
        for stim_num in stims:  # Обрабатывается каждый стимул для данного испытуемого
            for trial_num in trials:  # Обрабатывается каждая проба для данного стимула
                # Формируется ключ и путь к файлу в формате "SubX_StimX_TrialX.csv"
                eeg_key = f"Sub{sub_num}_Stim{stim_num}_Trial{trial_num}"
                input_file = os.path.join(data_dir, f"{eeg_key}.csv")  # Путь к файлу с данными

                # Проверка существования файла
                if not os.path.exists(input_file):
                    print(f"Файл {eeg_key}.csv не найден, пропуск...")
                    continue  # Если файл не найден, пропускается данный файл

                # Загружаются данные из CSV-файла
                eeg_df = pd.read_csv(input_file)

                # Проверяется наличие всех требуемых каналов
                missing_channels = [ch for ch in channels if ch not in eeg_df.columns]
                if missing_channels:
                    print(f"Каналы {missing_channels} не найдены в {eeg_key}.csv, пропуск...")
                    continue  # Если какие-то каналы отсутствуют, пропускается данный файл

                # Оставляются только те каналы, которые указаны в параметре channels
                eeg_df = eeg_df[channels]

                # Добавляются данные в словарь с ключом "SubX_StimX_TrialX"
                data_dict[eeg_key] = eeg_df

                print(f"Данные для {eeg_key} с выбранными каналами загружены.")

    return data_dict  # Возвращается словарь с загруженными данными

def organize(sub_num, stim_num, trial_num, data, channels=None):
    """
    Организуются данные для заданного испытуемого, стимула и пробы, загружаются данные и выбираются
    необходимые каналы. Если каналы не указаны, загружаются все столбцы.

    :param sub_num: Номер испытуемого.
    :param stim_num: Номер стимула.
    :param trial_num: Номер пробы.
    :param data: Путь к файлу данных.
    :param channels: Список каналов, которые нужно сохранить (опционально).
    :return: Словарь с загруженными данными.
    """
    data_dict = {}
    
    # Формируется ключ для данных
    eeg_key = f"Sub{sub_num}_Stim{stim_num}_Trial{trial_num}"
    input_file = data
    eeg_df = pd.read_csv(input_file)
    
    # Если каналы не переданы, используются все столбцы
    if channels is None:
        channels = eeg_df.columns.tolist()

    # Проверка наличия всех каналов в данных
    missing_channels = [ch for ch in channels if ch not in eeg_df.columns]
    
    if missing_channels:
        print(f"Каналы {missing_channels} не найдены в {eeg_key}, пропуск...")
        # Если каналы не найдены, выбираются только те, которые существуют
        eeg_df = eeg_df[[ch for ch in channels if ch in eeg_df.columns]]
    else:
        eeg_df = eeg_df[channels]

    # Добавление данных в словарь с соответствующим ключом
    data_dict[eeg_key] = eeg_df

    print(f"Данные для {eeg_key} с выбранными каналами загружены.")

    return data_dict  # Возвращается словарь с загруженными данными

def timeframe(data_dict, sampling_rate, start_time_sec, end_time_sec):
    """
    Выполняется выборка данных в пределах заданного временного интервала.
    Если данные в одной из записей недостаточны для выборки, работа прекращается.

    :param data_dict: Словарь, где ключ — это идентификатор (например, "Sub1_Trial3"),
                      а значение — DataFrame с данными.
    :param sampling_rate: Частота дискретизации данных (в Гц).
    :param start_time_sec: Время начала выборки в секундах.
    :param end_time_sec: Время конца выборки в секундах.
    :return: Новый словарь с данными, извлеченными в указанном временном интервале.
    """
    start_point = int(start_time_sec * sampling_rate)  # Перевод времени начала в количество точек
    end_point = int(end_time_sec * sampling_rate)  # Перевод времени конца в количество точек

    # Проверка корректности параметров
    if start_point >= end_point:
        raise ValueError("Время начала должно быть меньше времени конца.")
    
    selected_data_dict = {}  # Новый словарь для хранения выбранных данных

    # Обработка данных для каждого ключа в словаре
    for eeg_key, eeg_df in data_dict.items():
        eeg_data = eeg_df.values  # Преобразуются данные в массив numpy

        # Проверка на достаточность длины данных для выборки
        if len(eeg_data) < end_point:
            print(f"Данных для {eeg_key} недостаточно для выборки, работа прекращена.")
            return None  # Прекращается работа функции и возвращается None
        
        # Выборка данных в указанном временном интервале
        eeg_data_selected = eeg_data[start_point:end_point]

        # Создается DataFrame для выбранных данных
        selected_data_dict[eeg_key] = pd.DataFrame(eeg_data_selected, columns=eeg_df.columns)

        print(f"Данные для {eeg_key} успешно извлечены и добавлены в новый словарь.")

    return selected_data_dict  # Возвращается словарь с извлеченными данными

def average_trials_dynamic(data_dict):
    """
    Создается новый словарь, добавляются усредненные данные, динамически находя
    все пробы (Trial) для каждого субъекта (Sub) и стимула (Stim).
    Вычисляются все возможные попарные средние значения между существующими пробами
    одного стимула у одного субъекта, а также общее среднее всех проб.

    :param data_dict: Исходный словарь с данными. Ожидается, что ключи в этом словаре
                      имеют формат "SubX_StimY_TrialZ", где X, Y, Z - числовые идентификаторы.
                      Например: "Sub1_Stim1_Trial1", "Sub2_Stim3_Trial5".
    :return: **Новый словарь** с добавленными усредненными записями. Исходный словарь не изменяется.
    """
    # Создается поверхностная копия исходного словаря, чтобы не модифицировать оригинал
    new_data_dict = data_dict.copy()

    # Словарь для временной группировки данных: {'SubX_StimY': {trial_num: value, ...}}
    grouped_data = {} 
    
    # Регулярное выражение для парсинга ключей типа "Sub1_Stim1_Trial1"
    key_pattern = re.compile(r"^(Sub\d+)_(Stim\d+)_(Trial\d+)$") 

    # Шаг 1: Парсинг всех ключей в исходном словаре и группировка данных
    for key, value in new_data_dict.items():  # Итерируется по копии
        match = key_pattern.match(key)
        if match:
            subject_prefix = match.group(1)   # Например, "Sub1"
            stimulus_prefix = match.group(2)  # Например, "Stim1"
            trial_num = int(match.group(3).replace("Trial", "")) # Номер пробы, например, 1

            sub_stim_group_key = f"{subject_prefix}_{stimulus_prefix}"
            
            if sub_stim_group_key not in grouped_data:
                grouped_data[sub_stim_group_key] = {}
            grouped_data[sub_stim_group_key][trial_num] = value

    # Шаг 2: Обработка каждой группы "Субъект-Стимул", вычисление средних
    # и добавление новых записей в **новый** словарь.
    for sub_stim_prefix, trials_for_group in grouped_data.items():
        existing_trial_numbers = sorted(trials_for_group.keys())
        existing_values = [trials_for_group[tn] for tn in existing_trial_numbers]

        if len(existing_values) < 2: 
            if len(existing_values) == 1:
                print(f"\nДля группы **{sub_stim_prefix}** имеется только один триал (Trial{existing_trial_numbers[0]}). Попарные и общие средние не вычисляются.")
            else:
                print(f"\nДля группы **{sub_stim_prefix}** не найдено ни одного триала. Пропускается.")
            continue
        
        next_available_trial_num = max(existing_trial_numbers) + 1

        print(f"\n--- Обработка группы: **{sub_stim_prefix}** ---")
        print(f"Существующие триалы: {existing_trial_numbers}")

        # Вычисление всех возможных попарных средних значений
        for trial_num1, trial_num2 in combinations(existing_trial_numbers, 2):
            value1 = trials_for_group[trial_num1]
            value2 = trials_for_group[trial_num2]
            
            avg_pair = (value1 + value2) / 2
            
            new_key = f"{sub_stim_prefix}_Trial{next_available_trial_num}"
            new_data_dict[new_key] = avg_pair  # Добавляется в копию словаря
            print(f"  Добавлено: **{new_key}** (среднее между Trial{trial_num1} и Trial{trial_num2})") 
            next_available_trial_num += 1

        # Вычисление общего среднего значения для всех существующих триалов в группе
        overall_avg = sum(existing_values) / len(existing_values)
        new_key = f"{sub_stim_prefix}_Trial{next_available_trial_num}"
        new_data_dict[new_key] = overall_avg  # Добавляется в копию словаря
        print(f"  Добавлено: **{new_key}** (общее среднее всех {len(existing_values)} триалов)") 
        next_available_trial_num += 1

    return new_data_dict  # Возвращается новый словарь с усредненными данными

def bandpass_filter(data_dict, lowcut, highcut, fs, order=4):
    """
    Применяется полосовой фильтр Баттерворта ко всем данным в словаре.

    :param data_dict: Словарь, где ключом является идентификатор (например, "Sub1_Trial3"),
                      а значением — массив данных, подлежащих фильтрации.
    :param lowcut: Нижняя граница частотного диапазона фильтра (в Гц).
    :param highcut: Верхняя граница частотного диапазона фильтра (в Гц).
    :param fs: Частота дискретизации данных (в Гц).
    :param order: Порядок фильтра Баттерворта (по умолчанию 4).
    :return: Новый словарь, где каждый ключ сопоставляется с отфильтрованным массивом данных.
    """
    nyquist = 0.5 * fs  # Частота Найквиста (половина частоты дискретизации)
    low = lowcut / nyquist  # Нормализация нижней частоты
    high = highcut / nyquist  # Нормализация верхней частоты

    # Применяется фильтр Баттерворта для полосового пропускания
    b, a = signal.butter(order, [low, high], btype='band')

    # Новый словарь для хранения отфильтрованных данных
    filtered_data_dict = {}

    # Применяется фильтрация ко всем данным в словаре
    for key, data in data_dict.items():
        data_np = np.asarray(data)
        filtered_array = signal.filtfilt(b, a, data, axis=0)
        filtered_data_dict[key] = pd.DataFrame(filtered_array)
        print(f"Полосовой фильтр применен к записи {key}.")  # Сообщение о применении фильтра

    return filtered_data_dict  # Возвращается словарь с отфильтрованными данными 