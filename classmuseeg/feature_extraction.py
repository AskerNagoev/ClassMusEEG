import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import pywt

def moving_window(data_dict, window_size_sec, overlap_size_sec, sampling_rate):
    """
    Применяется скользящее окно к данным с учетом перекрытия, вычисляется среднее значение по каждому каналу.

    :param data_dict: Словарь, где ключом является идентификатор (например, "Sub1_Trial3"),
                      а значением — DataFrame с данными, к которым применяется скользящее окно.
    :param window_size_sec: Размер окна в секундах.
    :param overlap_size_sec: Размер перекрытия в секундах.
    :param sampling_rate: Частота дискретизации данных (в Гц).
    :return: Новый словарь, где ключом является тот же идентификатор, а значением — DataFrame с усредненными данными по окнам.
    """
    window_size = int(window_size_sec * sampling_rate)  # Перевод размера окна в количество точек
    overlap_size = int(overlap_size_sec * sampling_rate)  # Перевод размера перекрытия в количество точек

    # Проверка корректности параметров
    if window_size <= 0 or overlap_size < 0 or overlap_size >= window_size:
        raise ValueError("Некорректные параметры окна или перекрытия.")

    windowed_data_dict = {}  # Словарь для хранения обработанных данных

    # Обработка данных для каждого ключа в словаре
    for eeg_key, eeg_df in data_dict.items():
        if eeg_df.empty:
            print(f"Данные для {eeg_key} пустые, пропускаются...")
            continue

        eeg_data = eeg_df.values  # Преобразуются данные в массив numpy для обработки

        if len(eeg_data) < window_size:
            print(f"Данные для {eeg_key} слишком короткие для окна, пропускаются...")
            continue

        windowed_data = []  # Список для хранения усредненных данных по окнам

        # Применяется скользящее окно с перекрытием
        for start in range(0, len(eeg_data) - window_size + 1, window_size - overlap_size):
            end = start + window_size
            window = eeg_data[start:end]  # Извлекаются данные для текущего окна

            window_mean = np.mean(window, axis=0)  # Вычисляется среднее значение по каждому каналу в окне
            windowed_data.append(window_mean)  # Добавляется усредненное окно в список

        # Создается DataFrame для текущего ключа и добавляется в словарь
        windowed_df = pd.DataFrame(windowed_data, columns=eeg_df.columns)
        windowed_data_dict[eeg_key] = windowed_df

        print(f"Скользящее окно применено к {eeg_key}. Результат: {len(windowed_data)} окон.")

    return windowed_data_dict  # Возвращается словарь с данными после применения скользящего окна

def apply_window_and_fft(data_dict, sampling_rate, window_size_sec=1, overlap_sec=0.5, freq_range=(1, 7)):
    """
    Применяется скользящее окно и частотное преобразование (FFT) к данным из словаря.

    Для каждого окна данных применяется Быстрое Преобразование Фурье (FFT), и извлекаются амплитуды
    частот в заданном диапазоне. Результаты сохраняются в виде DataFrame с частотными компонентами.

    :param data_dict: Словарь, где ключ — это идентификатор записи (например, "Sub1_Trial1"), 
                      а значение — DataFrame с данными (временные ряды для разных каналов).
    :param sampling_rate: Частота дискретизации данных (в Гц).
    :param window_size_sec: Размер окна для скользящего окна в секундах.
    :param overlap_sec: Перекрытие окон в секундах.
    :param freq_range: Диапазон частот (в Гц) для извлечения частотных компонент.
    :return: Словарь с обработанными данными для каждого идентификатора записи, 
             где значения — это DataFrame с амплитудами частот.
    """
    # Переводятся размер окна и перекрытия в количество точек
    window_size = int(window_size_sec * sampling_rate)
    overlap_size = int(overlap_sec * sampling_rate)
    step_size = window_size - overlap_size  # Шаг для скользящего окна

    # Новый словарь для хранения результатов обработки
    processed_data_dict = {}

    # Процесс обработки данных для каждого идентификатора записи в словаре
    for key, df in data_dict.items():
        # Преобразуются DataFrame в массив numpy для удобства обработки
        data = df.values  # Массив с данными для каждого канала
        num_channels = data.shape[1]  # Количество каналов

        # Список для хранения частотных признаков (амплитуд) для каждого окна
        freq_features = []

        # Применяется скользящее окно к данным
        for start in range(0, len(data) - window_size + 1, step_size):
            end = start + window_size
            window = data[start:end, :]  # Вырезаются данные для текущего окна

            # Применяется FFT для каждого канала в окне
            window_freq_features = []
            for channel in range(num_channels):
                # Выполняется FFT для текущего канала
                fft_result = fft(window[:, channel])
                freqs = fftfreq(window_size, 1 / sampling_rate)  # Частотная ось

                # Фильтруются частоты по заданному диапазону
                mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                fft_magnitude = np.abs(fft_result[mask])  # Амплитуды в заданном диапазоне частот

                # Добавляются амплитуды текущего канала к результатам окна
                window_freq_features.extend(fft_magnitude)

            # Добавляются извлечённые частотные компоненты для текущего окна
            freq_features.append(window_freq_features)

        # Преобразуются список амплитуд в DataFrame для текущей записи
        freq_features_df = pd.DataFrame(freq_features)

        # Сохраняется DataFrame с частотными компонентами в результирующий словарь
        processed_data_dict[key] = freq_features_df

        # Печать информации о текущей записи
        print(f"Обработана запись {key}. Формат данных: {freq_features_df.shape}")

    # Возвращается словарь с результатами обработки для всех записей
    return processed_data_dict

def apply_windowed_wavelet_transform(
    data_dict, sampling_rate, window_size_sec=1, overlap_sec=0.5, freq_range=(1, 7), num_freqs=10
):
    """
    Применяется оконное временно-частотное преобразование с использованием вейвлет-преобразования для всех каналов.
    Возвращается словарь, где ключи — это имена записей, а значения — DataFrame с объединенными данными для всех каналов.
    
    :param data_dict: Словарь, где ключ — это идентификатор записи (например, "Sub1_Trial1"),
                      а значение — это DataFrame с данными для каждого канала.
    :param sampling_rate: Частота дискретизации данных (в Гц).
    :param window_size_sec: Размер окна для скользящего окна в секундах.
    :param overlap_sec: Перекрытие окон в секундах.
    :param freq_range: Диапазон частот для анализа (в Гц).
    :param num_freqs: Количество частот для преобразования.
    :return: Словарь с обработанными данными для каждой записи.
    """
    # Параметры окна
    window_size = int(window_size_sec * sampling_rate)  # Размер окна в точках
    overlap_size = int(overlap_sec * sampling_rate)  # Размер перекрытия в точках
    step_size = window_size - overlap_size  # Шаг скользящего окна

    # Частоты для преобразования (линейно распределены по заданному диапазону)
    frequencies = np.linspace(freq_range[0], freq_range[1], num=num_freqs)

    # Преобразование частот в масштабы (widths)
    central_freq = 1.0  # Центральная частота для вейвлета Морле
    scales = central_freq * sampling_rate / frequencies  # Масштабы для вейвлет-преобразования

    # Словарь для хранения результатов
    transformed_data_dict = {}

    def compute_wavelet(data, scales):
        """Выполняется вейвлет-преобразование с использованием Морле."""
        return pywt.cwt(data, scales, 'morl')[0]  # Преобразование с использованием вейвлета Морле

    # Обработка данных для каждой записи в словаре
    for key, df in data_dict.items():
        channels = df.columns.tolist()  # Получаются каналы
        all_data = []  # Список для хранения данных для всех каналов

        # Обработка каждого канала данных
        for channel in channels:
            channel_data = df[channel].values  # Получаются данные для текущего канала
            transformed_data = []  # Список для хранения преобразованных данных

            # Применяется скользящее окно
            for start in range(0, len(channel_data) - window_size + 1, step_size):
                end = start + window_size
                window_data = channel_data[start:end]  # Вырезаются данные для окна

                # Выполняется вейвлет-преобразование для окна
                wavelet_transform = compute_wavelet(window_data, scales)

                # Вычисляется среднее значение амплитуды по каждому окну
                mean_transform = np.mean(np.abs(wavelet_transform), axis=1)
                transformed_data.append(mean_transform)  # Добавляются результаты преобразования

            # Добавляются данные для текущего канала в общий список
            all_data.append(np.array(transformed_data))

        # Объединяются данные для всех каналов в один DataFrame
        combined_data = np.hstack(all_data)  # Объединяются данные всех каналов
        columns = [f"{channel}_Freq_{f:.1f}Hz" for channel in channels for f in frequencies]  # Названия столбцов
        transformed_data_dict[key] = pd.DataFrame(combined_data, columns=columns)  # Создается DataFrame для текущей записи

        print(f"Обработана запись {key}. Формат данных: {combined_data.shape}")  # Печать информации о данных

    return transformed_data_dict  # Возвращается словарь с результатами

def column_wise_mean(data_dict):
    """
    Применяется усреднение значений по столбцам для каждой записи в словаре.

    :param data_dict: Словарь, где ключом является идентификатор (например, "Sub1_Trial3"),
                      а значением — DataFrame с данными.
    :return: Новый словарь, где ключом является тот же идентификатор, а значением — вектор средних значений по каждому столбцу.
    """
    mean_data_dict = {}  # Словарь для хранения усредненных данных

    # Обработка данных для каждого ключа в словаре
    for eeg_key, eeg_df in data_dict.items():
        if eeg_df.empty:
            print(f"Данные для {eeg_key} пустые, пропускаются...")
            continue

        # Усреднение по каждому столбцу
        column_means = eeg_df.mean(axis=0).values  # Среднее значение по каждому столбцу
        mean_data_dict[eeg_key] = column_means  # Сохранение усредненного вектора

        print(f"Усреднены значения для {eeg_key}. Результат: {len(column_means)} значений.")

    return mean_data_dict  # Возвращается словарь с усредненными данными 