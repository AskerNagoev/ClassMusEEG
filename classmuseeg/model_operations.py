import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import random
import tensorflow as tf

def process_data(data_dict, subjects, stims):
    """
    Обрабатываются данные, извлекаются записи для заданных испытуемых и стимулов,
    а также разделяются данные на обучающую, валидационную и тестовую выборки.
    
    Функция создает два списка: X для признаков и y для меток, где каждый элемент
    соответствует одному эксперименту. Также создается маппинг стимулов на метки,
    после чего данные разделяются на обучающую, валидационную и тестовую выборки.
    
    :param data_dict: Словарь, где ключом является строка с идентификатором испытуемого и стимула,
                      а значением — данные, соответствующие данному эксперименту.
    :param subjects: Список с идентификаторами испытуемых, данные которых будут выбраны.
    :param stims: Список с номерами стимулов, данные для которых будут выбраны.
    :return: Четыре массива для разделенных данных: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # Создаются списки для признаков X и меток y
    X = []
    y = []
    
    # Множество для хранения уникальных стимулов
    unique_stimuli = set()
    
    # Итерируется по ключам словаря данных и выбираются данные для указанных стимулов и испытуемых
    for key, value in data_dict.items():
        # Извлекаются номер испытуемого и стимула из ключа
        subject_id = int(key.split('_')[0][3:])  # извлекается номер испытуемого
        stimulus = int(key.split('_')[1][4:])  # извлекается номер стимула

        # Проверяется наличие испытуемого и стимула в заданных списках
        if subject_id in subjects and stimulus in stims:
            X.append(value)  # Добавляются данные в список признаков
            unique_stimuli.add(stimulus)  # Добавляется стимул в множество уникальных стимулов

    # Создается маппинг стимулов на метки (сортировка стимулов и присвоение меток от 0)
    stimulus_to_label = {stimulus: idx for idx, stimulus in enumerate(sorted(unique_stimuli))}
    
    # Присваиваются метки y, соответствующие стимулу каждого ключа
    y = [stimulus_to_label[int(key.split('_')[1][4:])] for key, value in data_dict.items() 
         if int(key.split('_')[1][4:]) in stimulus_to_label and int(key.split('_')[0][3:]) in subjects]
    
    # Преобразуются X и y в массивы NumPy для дальнейшего использования в train_test_split
    X = np.array(X)
    y = np.array(y)
    
    # Разделяются данные на обучающую, валидационную и тестовую выборки
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Выводится информация о размере выборок
    print("Данные для выбранных стимулов и субъектов:")
    print("Размер обучающей выборки X_train:", X_train.shape)
    print("Размер валидационной выборки X_val:", X_val.shape)
    print("Размер тестовой выборки X_test:", X_test.shape)
    
    def check_class_distribution(y, set_name):
        """
        Проверяется распределение классов в заданной выборке и выводится статистика.
        
        :param y: Массив меток классов.
        :param set_name: Название выборки (например, "обучающая выборка").
        """
        unique, counts = np.unique(y, return_counts=True)
        print(f"Распределение классов в {set_name}:")
        for cls, count in zip(unique, counts):
            print(f"Класс {cls}: {count} примеров ({count / len(y) * 100:.2f}%)")
    
    # Проверяется распределение классов в обучающей, валидационной и тестовой выборках
    check_class_distribution(y_train, "обучающей выборке")
    check_class_distribution(y_val, "валидационной выборке")
    check_class_distribution(y_test, "тестовой выборке")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_channels(X_train, X_val, X_test):
    """
    Применяется нормализация всех каналов для обучающего, валидационного и тестового наборов данных.

    :param X_train: Обучающий набор данных с формой (n_samples_train, timesteps, n_features).
    :param X_val: Валидационный набор данных с формой (n_samples_val, timesteps, n_features).
    :param X_test: Тестовый набор данных с формой (n_samples_test, timesteps, n_features).
    :return: Нормализованные обучающий, валидационный и тестовый наборы данных.
    """
    # Преобразуются данные в два измерения для нормализации всех каналов одновременно
    n_samples_train, timesteps, n_features = X_train.shape
    
    # Переводятся данные в форму (n_samples * timesteps, n_features), чтобы применить масштабирование ко всем каналам
    X_train_reshaped = X_train.reshape(n_samples_train * timesteps, n_features)
    
    # Инициализируется scaler
    scaler = StandardScaler()
    
    # Применяется нормализация ко всем данным из обучающего набора
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    
    # Для валидационных и тестовых данных используется только transform
    n_samples_val, _, _ = X_val.shape
    X_val_reshaped = X_val.reshape(n_samples_val * timesteps, n_features)
    X_val_scaled = scaler.transform(X_val_reshaped)
    
    n_samples_test, _, _ = X_test.shape
    X_test_reshaped = X_test.reshape(n_samples_test * timesteps, n_features)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Возвращаются данные в исходную форму (n_samples, timesteps, n_features)
    X_train_normalized = X_train_scaled.reshape(n_samples_train, timesteps, n_features)
    X_val_normalized = X_val_scaled.reshape(n_samples_val, timesteps, n_features)
    X_test_normalized = X_test_scaled.reshape(n_samples_test, timesteps, n_features)
    
    return X_train_normalized, X_val_normalized, X_test_normalized, scaler

def optimize_model_lstm(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=100):
    """
    Оптимизируется модель LSTM с помощью Optuna для поиска лучших гиперпараметров.
    Модели обучаются с использованием обучающей выборки, затем проводится оценка 
    на валидационной и тестовой выборках.

    :param X_train: Обучающий набор данных.
    :param y_train: Метки для обучающего набора данных.
    :param X_val: Валидационный набор данных.
    :param y_val: Метки для валидационного набора данных.
    :param X_test: Тестовый набор данных.
    :param y_test: Метки для тестового набора данных.
    :param n_trials: Количество проб в процессе оптимизации гиперпараметров.
    :return: DataFrame с результатами и словарь с сохраненными пробами.
    """
    # Устанавливаются фиксированные значения для случайных чисел для воспроизводимости
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    # Списки для хранения результатов и параметров гиперпараметров
    study_results = []
    saved_trials = {}

    def create_model(trial):
        """
        Создается модель LSTM с гиперпараметрами, предложенными Optuna.
        
        :param trial: Текущий проба гиперпараметров.
        :return: Скомпилированная модель.
        """
        model = Sequential()

        # Количество LSTM слоев выбирается на основе гиперпараметров
        n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [1, 2, 3, 4, 5])

        # Добавляются LSTM слои
        for i in range(n_lstm_layers):
            units = trial.suggest_categorical(f"lstm_units_layer_{i}", [8, 16, 32, 64, 128, 256])
            return_sequences = (i < n_lstm_layers - 1)
        
            if i == 0:
                model.add(LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizers.l2(trial.suggest_categorical("l2_regularization", [0.1, 0.01, 0.001])),
                    input_shape=(X_train.shape[1], X_train.shape[2])
                ))
            else:
                model.add(LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizers.l2(trial.suggest_categorical("l2_regularization", [0.1, 0.01, 0.001]))
                ))
            
            # Добавляется Layer Normalization
            model.add(LayerNormalization())
        
            # Добавляется Dropout
            dropout_rate = trial.suggest_categorical(f"dropout_rate_{i}", [0, 0.2, 0.3, 0.4, 0.5])
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        # Количество Dense слоев выбирается на основе гиперпараметров
        n_dense_layers = trial.suggest_categorical("n_dense_layers", [1, 2, 3])

        # Добавляются Dense слои
        for i in range(n_dense_layers):
            dense_units = trial.suggest_categorical(f"dense_units_{i}", [8, 16, 32, 64, 128, 256])
            model.add(Dense(dense_units, activation='relu'))

            dropout_rate = trial.suggest_categorical(f"dense_dropout_rate_{i}", [0, 0.2, 0.3, 0.4, 0.5])
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        # Финальный слой для классификации с мняющимся количеством выходов
        num_classes = len(np.unique(y_train))
        model.add(Dense(num_classes, activation='softmax'))  

        # Выбор оптимизатора и начальной скорости обучения
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        learning_rate = trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.01])

        if optimizer_name == "Adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == "RMSprop":
            optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

        # Компиляция модели
        model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        return model

    def objective(trial):
        """
        Определяется цель для оптимизации с помощью Optuna. Модель обучается и оценивается на трех выборках.
        
        :param trial: Текущий проба гиперпараметров.
        :return: Значение для минимизации (потеря на валидации).
        """
        model = create_model(trial)

        # Прерывание обучения при отсутствии улучшений на валидационной выборке
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            min_delta=0.0001,
            restore_best_weights=True
        )

        # Обучение модели
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1000,
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            callbacks=[early_stopping],
            verbose=0
        )

        # Оценка модели на обучающей, валидационной и тестовой выборках
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        # Сохранение результатов для каждого пробы
        study_results.append({
            'Trial': trial.number,
            'Train Accuracy': train_accuracy,
            'Validation Accuracy': val_accuracy,
            'Validation Loss': val_loss,
            'Test Accuracy': test_accuracy
        })

        saved_trials[trial.number] = trial.params

        # Сохранение весов модели для дальнейшего анализа
        model.save_weights(f'model_weights_trial_{trial.number}.weights.h5')
        
        print(f"Trial {trial.number}: Train Accuracy={train_accuracy:.4f}, Validation Accuracy={val_accuracy:.4f}, Test Accuracy={test_accuracy:.4f}")

        return val_loss

    # Создание исследования для минимизации потери на валидации
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Вывод наилучших гиперпараметров
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)

    # Возвращение результатов исследования
    results_df = pd.DataFrame(study_results)

    return results_df, saved_trials

def optimize_model_bilstm(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=100):
    """
    Оптимизируется модель BiLSTM с помощью Optuna для поиска лучших гиперпараметров.
    Модели обучаются с использованием обучающей выборки, затем проводится оценка 
    на валидационной и тестовой выборках.

    :param X_train: Обучающий набор данных.
    :param y_train: Метки для обучающего набора данных.
    :param X_val: Валидационный набор данных.
    :param y_val: Метки для валидационного набора данных.
    :param X_test: Тестовый набор данных.
    :param y_test: Метки для тестового набора данных.
    :param n_trials: Количество проб в процессе оптимизации гиперпараметров.
    :return: DataFrame с результатами и словарь с сохраненными пробами.
    """
    # Устанавливаются фиксированные значения для случайных чисел для воспроизводимости
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    # Списки для хранения результатов и параметров гиперпараметров
    study_results = []
    saved_trials = {}

    def create_model(trial):
        """
        Создается модель BiLSTM с гиперпараметрами, предложенными Optuna.
        
        :param trial: Текущий проба гиперпараметров.
        :return: Скомпилированная модель.
        """
        model = Sequential()

        # Количество BiLSTM слоев выбирается на основе гиперпараметров
        n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [1, 2, 3, 4, 5])

        # Добавляются BiLSTM слои
        for i in range(n_lstm_layers):
            units = trial.suggest_categorical(f"lstm_units_layer_{i}", [8, 16, 32, 64, 128, 256])
            return_sequences = (i < n_lstm_layers - 1)

            # Используется Bidirectional обертка для LSTM
            if i == 0:
                model.add(Bidirectional(
                    LSTM(
                        units,
                        return_sequences=return_sequences,
                        kernel_regularizer=regularizers.l2(trial.suggest_categorical("l2_regularization", [0.1, 0.01, 0.001])),
                        input_shape=(X_train.shape[1], X_train.shape[2])
                    )
                ))
            else:
                model.add(Bidirectional(
                    LSTM(
                        units,
                        return_sequences=return_sequences,
                        kernel_regularizer=regularizers.l2(trial.suggest_categorical("l2_regularization", [0.1, 0.01, 0.001]))
                    )
                ))
            
            # Добавляется Layer Normalization
            model.add(LayerNormalization())
        
            # Добавляется Dropout
            dropout_rate = trial.suggest_categorical(f"dropout_rate_{i}", [0, 0.2, 0.3, 0.4, 0.5])
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        # Количество Dense слоев выбирается на основе гиперпараметров
        n_dense_layers = trial.suggest_categorical("n_dense_layers", [1, 2, 3])

        # Добавляются Dense слои
        for i in range(n_dense_layers):
            dense_units = trial.suggest_categorical(f"dense_units_{i}", [8, 16, 32, 64, 128, 256])
            model.add(Dense(dense_units, activation='relu'))

            dropout_rate = trial.suggest_categorical(f"dense_dropout_rate_{i}", [0, 0.2, 0.3, 0.4, 0.5])
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        # Финальный слой для классификации с мняющимся количеством выходов
        num_classes = len(np.unique(y_train))
        model.add(Dense(num_classes, activation='softmax'))

        # Выбор оптимизатора и начальной скорости обучения
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        learning_rate = trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.01])

        if optimizer_name == "Adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == "RMSprop":
            optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

        # Компиляция модели
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def objective(trial):
        """
        Определяется цель для оптимизации с помощью Optuna. Модель обучается и оценивается на трех выборках.
        
        :param trial: Текущий проба гиперпараметров.
        :return: Значение для минимизации (потеря на валидации).
        """
        model = create_model(trial)

        # Прерывание обучения при отсутствии улучшений на валидационной выборке
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            min_delta=0.0001,
            restore_best_weights=True
        )

        # Обучение модели
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1000,
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            callbacks=[early_stopping],
            verbose=0
        )

        # Оценка модели на обучающей, валидационной и тестовой выборках
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        # Сохранение результатов для каждого пробы
        study_results.append({
            'Trial': trial.number,
            'Train Accuracy': train_accuracy,
            'Validation Accuracy': val_accuracy,
            'Validation Loss': val_loss,
            'Test Accuracy': test_accuracy
        })

        saved_trials[trial.number] = trial.params

        # Сохранение весов модели для дальнейшего анализа
        model.save_weights(f'model_weights_trial_{trial.number}.weights.h5')
        
        print(f"Trial {trial.number}: Train Accuracy={train_accuracy:.4f}, Validation Accuracy={val_accuracy:.4f}, Test Accuracy={test_accuracy:.4f}")

        return val_loss

    # Создание исследования для минимизации потери на валидации
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Вывод наилучших гиперпараметров
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)

    # Возвращение результатов исследования
    results_df = pd.DataFrame(study_results)

    return results_df, saved_trials

def optimize_model_svm(X_train, X_val, X_test, y_train, y_val, y_test, n_trials=100):
    """
    Оптимизируется модель классификатора SVM (Support Vector Machine)
    с использованием Optuna для выбора гиперпараметров.
    Включается стандартизация данных, использование Optuna для выбора гиперпараметров и оценка производительности модели.

    :param X_train: Признаки обучающей выборки
    :param X_val: Признаки валидационной выборки
    :param X_test: Признаки тестовой выборки
    :param y_train: Метки обучающей выборки
    :param y_val: Метки валидационной выборки
    :param y_test: Метки тестовой выборки
    :param n_trials: Количество итераций для оптимизации
    :return: scaler: Обученный скейлер для дальнейшего использования
             best_model: Лучшая обученная модель с оптимизированными гиперпараметрами
    """
    # Стандартизируются данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    def objective(trial):
        """
        Определяется целевая функция для Optuna, которая выбирает гиперпараметры для SVC,
        обучает модель и возвращает метрику для оптимизации (точность на валидационной выборке).
        
        :param trial: Текущий проба гиперпараметров.
        :return: Точность модели на валидационной выборке.
        """
        # Определяются базовые гиперпараметры для SVC
        svc_params = {
            'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'random_state': 42
        }

        # Гиперпараметры, зависящие от выбранного типа ядра (kernel)
        if svc_params['kernel'] == 'poly':
            svc_params['degree'] = trial.suggest_int('degree', 2, 5)
            svc_params['gamma'] = trial.suggest_float('gamma', 1e-3, 1, log=True)
            svc_params['coef0'] = trial.suggest_float('coef0', -1.0, 1.0)
        elif svc_params['kernel'] in ['rbf', 'sigmoid']:
            svc_params['gamma'] = trial.suggest_float('gamma', 1e-3, 1, log=True)
            if svc_params['kernel'] == 'sigmoid':
                svc_params['coef0'] = trial.suggest_float('coef0', -1.0, 1.0)

        # Создается модель SVC с предложенными гиперпараметрами
        model = SVC(probability=True, **svc_params)  # Включаем probability=True для использования predict_proba

        # Обучение модели на обучающих данных
        model.fit(X_train_scaled, y_train)

        # Оценка модели на валидационной выборке (используем predict_proba для вычисления log_loss)
        y_val_prob = model.predict_proba(X_val_scaled)
        val_loss = log_loss(y_val, y_val_prob)

        return val_loss

    # Создается исследование Optuna и запускается оптимизация
    study = optuna.create_study(direction='minimize', study_name='SVM_Optimization')  # Минимизируем потери
    
    print("Запуск оптимизации гиперпараметров SVM с Optuna...")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # Используем n_jobs=-1 для многозадачности

    # Получаются лучшие параметры, найденные Optuna
    best_params = study.best_trial.params
    print(f"\nЛучшие параметры, найденные Optuna: {best_params}")

    # Получается обученная модель с лучшими гиперпараметрами
    best_model_params = best_params.copy()
    best_model = SVC(probability=True, random_state=42, **best_model_params)

    # Обучение лучшей модели на всей обучающей выборке
    print("\nОбучение финальной модели SVM с лучшими параметрами...")
    best_model.fit(X_train_scaled, y_train)

    # Оценка лучшей модели на валидационной выборке
    y_val_pred = best_model.predict(X_val_scaled)
    print('\nОтчет по классификации на валидационной выборке (с лучшей моделью, обученной на X_train):')
    print(classification_report(y_val, y_val_pred))

    # Выводится точность на валидационной выборке
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy на валидационной выборке: {val_accuracy:.6f}")

    # Оценка лучшей модели на тестовой выборке
    y_test_pred = best_model.predict(X_test_scaled)
    print('\nОтчет по классификации на тестовой выборке:')
    print(classification_report(y_test, y_test_pred))

    # Выводится значение accuracy для тестовой выборки
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy на тестовой выборке: {test_accuracy:.6f}")

    return best_model, scaler

def optimize_model_sgd(X_train, X_val, X_test, y_train, y_val, y_test, n_trials=100):
    """
    Оптимизируется модель линейного классификатора с помощью стохастического градиентного спуска (SGD)
    с использованием Optuna для выбора гиперпараметров.
    Включается стандартизация данных, использование Optuna для выбора гиперпараметров и оценка производительности модели.

    :param X_train: Признаки обучающей выборки
    :param X_val: Признаки валидационной выборки
    :param X_test: Признаки тестовой выборки
    :param y_train: Метки обучающей выборки
    :param y_val: Метки валидационной выборки
    :param y_test: Метки тестовой выборки
    :param n_trials: Количество итераций для оптимизации
    :return: best_model: Лучшая обученная модель с оптимизированными гиперпараметрами
             scaler: Обученный скейлер для дальнейшего использования
    """
    # Стандартизация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    def objective(trial):
        """
        Целевая функция для Optuna, которая выбирает гиперпараметры для SGDClassifier,
        обучает модель и возвращает метрику для оптимизации (потери на валидационной выборке).
        
        :param trial: Текущий проба гиперпараметров.
        :return: Потери модели на валидационной выборке (log_loss).
        """
        # Ограничиваем выбор loss теми, которые поддерживают predict_proba
        loss = trial.suggest_categorical('loss', ['log_loss', 'modified_huber'])

        alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
        max_iter = trial.suggest_int('max_iter', 1000, 10000, step=1000)
        tol = trial.suggest_float('tol', 1e-4, 1e-2, log=True)
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
        early_stopping = trial.suggest_categorical('early_stopping', [True, False])

        # eta0 применяется только для некоторых стратегий learning_rate
        eta0 = None
        if learning_rate in ['constant', 'invscaling', 'adaptive']:
            eta0 = trial.suggest_float('eta0', 0.001, 0.1, log=True)

        # Создается модель стохастического градиентного спуска с предложенными гиперпараметрами
        model_params = {
            'loss': loss,
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol,
            'learning_rate': learning_rate,
            'penalty': penalty,
            'early_stopping': early_stopping,
            'random_state': 42
        }
        if eta0 is not None:
            model_params['eta0'] = eta0

        model = SGDClassifier(**model_params)

        # Обучение модели на обучающей выборке
        model.fit(X_train_scaled, y_train)

        # Оценка модели на валидационной выборке (используем predict_proba для вычисления log_loss)
        try:
            y_val_prob = model.predict_proba(X_val_scaled)  # Получаем вероятности
            val_loss = log_loss(y_val, y_val_prob)  # Рассчитываем потери
        except AttributeError:
            # Если модель не поддерживает predict_proba, просто пропустим её
            return float('inf')

        return val_loss

    # Создается исследование Optuna и запускается оптимизация
    study = optuna.create_study(direction='minimize', study_name='SGDClassifier_Optimization')  # Минимизируем потери
    
    print("Запуск оптимизации гиперпараметров с Optuna...")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    # Получаются лучшие параметры, найденные Optuna
    best_params = study.best_trial.params
    print(f"\nЛучшие параметры, найденные Optuna: {best_params}")

    # Создается и обучается лучшая модель с найденными гиперпараметрами
    best_model_params = best_params.copy()
    best_model = SGDClassifier(random_state=42, **best_model_params)

    # Обучение лучшей модели на всей обучающей выборке
    print("\nОбучение финальной модели с лучшими параметрами...")
    best_model.fit(X_train_scaled, y_train)

    # Оценка лучшей модели на валидационной выборке
    y_val_pred = best_model.predict(X_val_scaled)
    print('\nОтчет по классификации на валидационной выборке (с лучшей моделью, обученной на X_train):')
    print(classification_report(y_val, y_val_pred))

    # Выводится точность на валидационной выборке
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy на валидационной выборке: {val_accuracy:.6f}")

    # Оценка лучшей модели на тестовой выборке
    y_test_pred = best_model.predict(X_test_scaled)
    print('\nОтчет по классификации на тестовой выборке:')
    print(classification_report(y_test, y_test_pred))

    # Выводится значение accuracy для тестовой выборки
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy на тестовой выборке: {test_accuracy:.6f}")

    return best_model, scaler

def recreate_model(params, weights_file=None, freeze_layers=True, X_train=None, model_type="LSTM", n_classes=None):
    """
    Восстанавливается модель с гиперпараметрами, указанными в параметрах.
    Веса модели загружаются, если указан файл с весами. Также есть возможность
    заморозить слои модели, чтобы они не обновлялись при обучении.

    :param params: Гиперпараметры, которые будут использоваться для создания модели.
    :param weights_file: Файл с весами модели, которые будут загружены.
    :param freeze_layers: Если True, все слои, кроме последних, будут заморожены.
    :param X_train: Признаки обучающей выборки.
    :param model_type: Тип модели: "LSTM" или "Bi-LSTM".
    :param n_classes: Количество классов для классификации. Если None, будет вычислено из X_train.
    :return: Восстановленная модель.
    """
    model = Sequential()

    # Количество LSTM слоев выбирается на основе гиперпараметров
    n_lstm_layers = params['n_lstm_layers']
    for i in range(n_lstm_layers):
        units = params[f"lstm_units_layer_{i}"]
        return_sequences = (i < n_lstm_layers - 1)
    
        # Выбор между LSTM или Bi-LSTM
        if model_type == "Bi-LSTM":
            # Добавляем Bidirectional LSTM слой
            if i == 0:
                model.add(Bidirectional(LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizers.l2(params["l2_regularization"]),
                    input_shape=(X_train.shape[1], X_train.shape[2])
                )))
            else:
                model.add(Bidirectional(LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizers.l2(params["l2_regularization"])
                )))
        else:
            # Добавляем обычный LSTM слой
            if i == 0:
                model.add(LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizers.l2(params["l2_regularization"]),
                    input_shape=(X_train.shape[1], X_train.shape[2])
                ))
            else:
                model.add(LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizers.l2(params["l2_regularization"])
                ))

        model.add(LayerNormalization())

        dropout_rate = params[f"dropout_rate_{i}"]
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Количество Dense слоев выбирается на основе гиперпараметров
    n_dense_layers = params['n_dense_layers']
    for i in range(n_dense_layers):
        dense_units = params[f"dense_units_{i}"]
        model.add(Dense(dense_units, activation='relu'))

        dropout_rate = params[f"dense_dropout_rate_{i}"]
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Определяем количество классов, если параметр не передан
    if n_classes is None:
        raise ValueError("Количество классов (n_classes) должно быть передано.")

    # Финальный слой для классификации с количеством выходов, соответствующим количеству классов
    model.add(Dense(n_classes, activation='softmax'))  

    # Выбор оптимизатора и начальной скорости обучения
    optimizer_name = params["optimizer"]
    learning_rate = params["learning_rate"]

    if optimizer_name == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

    # Компиляция модели
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Строим модель с входными данными
    model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))

    # Восстанавливаются веса, если файл с весами предоставлен
    if weights_file:
        model.load_weights(weights_file)

    # Замораживаются все слои, кроме последних
    if freeze_layers:
        for layer in model.layers[:-1]:
            layer.trainable = False

    return model

def matrix_confusion(model, X_test, y_test):
    """
    Строится и визуализируется матрица ошибок для заданной модели на тестовых данных.

    :param model: Восстановленная или обученная модель для предсказаний.
    :param X_test: Тестовые данные (признаки).
    :param y_test: Истинные метки классов для тестовых данных.
    """
    # Получаются предсказания на тестовых данных
    y_pred = model.predict(X_test)

    # Если модель возвращает вероятности (например, для многоклассовой классификации), извлекаем метки
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:  # если y_pred - это вероятности
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:  # если модель уже возвращает метки классов
        y_pred_labels = y_pred

    # Вычисляется количество классов (на основе данных или предсказаний)
    num_classes = len(np.unique(np.concatenate((y_test, y_pred_labels))))

    # Строится матрица ошибок
    cm = confusion_matrix(y_test, y_pred_labels)

    # Визуализируется матрица ошибок
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))

    # Настроиваются подписи
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок')
    plt.show()
