import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

class WindForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Прогноз скорости ветра")
        self.root.geometry("1200x900")
        self.root.configure(bg="#f0f0f0")

        # Уменьшение шрифта на графиках в 3 раза
        default_font_size = matplotlib.rcParams['font.size']
        matplotlib.rcParams.update({'font.size': default_font_size / 3})

        # Стиль для кнопок
        button_style = {
            "font": ("Arial", 12, "bold"),
            "bg": "#4CAF50",
            "fg": "white",
            "activebackground": "#45a049",
            "activeforeground": "white",
            "relief": "flat",
            "padx": 15,
            "pady": 5
        }

        # Основной фрейм для разделения интерфейса
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill="both", expand=True)

        # Левый фрейм для требований
        self.left_frame = tk.Frame(self.main_frame, width=200, bg="#e0e0e0")
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Требования к CSV-файлу
        requirements_text = (
            "Требования к CSV-файлу:\n\n"
            "1. Формат: CSV\n"
            "2. Столбцы:\n"
            "   - Date: дата в формате YYYYMMDD\n"
            "   - wind_on_10meters: скорость ветра (м/с)\n"
            "   - temp: температура (°C)\n"
            "3. Данные должны быть последовательными\n"
            "4. Минимум 72 часа данных для прогноза"
        )
        self.requirements_label = tk.Label(
            self.left_frame, text=requirements_text, font=("Arial", 10), bg="#e0e0e0", justify="left", anchor="nw"
        )
        self.requirements_label.pack(pady=10, padx=10)

        # Правый фрейм для основного контента
        self.right_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Кнопка для загрузки файла
        self.load_button = tk.Button(self.right_frame, text="Загрузить файл", command=self.load_file, **button_style)
        self.load_button.pack(pady=10)

        # Кнопка для запуска прогноза
        self.forecast_button = tk.Button(self.right_frame, text="Сделать прогноз", command=self.run_forecast, state=tk.DISABLED, **button_style)
        self.forecast_button.pack(pady=5)

        # Метка для статуса
        self.status_label = tk.Label(self.right_frame, text="Ожидание загрузки файла...", font=("Arial", 12), bg="#f0f0f0")
        self.status_label.pack(pady=10)

        # Текстовое поле для вывода процесса обучения
        self.log_text = scrolledtext.ScrolledText(self.right_frame, height=5, width=80, font=("Arial", 10), state='disabled')
        self.log_text.pack(pady=10)

        # Область для графиков (вертикальное расположение)
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(pady=10)

        # Метка для отображения достоверности
        self.accuracy_label = tk.Label(self.right_frame, text="Достоверность прогноза: Ожидание прогноза...", font=("Arial", 12), bg="#f0f0f0")
        self.accuracy_label.pack(pady=10)

        # Переменные
        self.df = None
        self.model = None
        self.scalers = {}
        self.history = None
        self.test_timestamps = None

    def log_message(self, message):
        """Функция для вывода сообщений в текстовое поле"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update()

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.df = self.df.drop(['Unnamed: 0', '10u', '10v'], axis=1, errors='ignore')
                self.status_label.config(text="Файл загружен. Подготовка данных...")
                self.log_message("Файл загружен: " + file_path)
                self.preprocess_data()
                self.forecast_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")
                self.status_label.config(text="Ошибка при загрузке файла")
                self.log_message(f"Ошибка: {e}")

    def preprocess_data(self):
        # Парсинг дат
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y%m%d')
        datetimes = []
        for date in self.df['Date'].unique():
            day_records = self.df[self.df['Date'] == date]
            num_records = len(day_records)
            hours_increment = 24 / num_records if num_records > 0 else 1
            for i in range(num_records):
                hour = i * hours_increment
                dt = date + timedelta(hours=hour)
                datetimes.append(dt)

        assert len(datetimes) == len(self.df), "Несоответствие длины datetimes и DataFrame"
        self.df['datetime'] = datetimes
        self.df = self.df.set_index('datetime')
        self.df = self.df.drop('Date', axis=1)

        # Обработка дубликатов
        if self.df.index.duplicated().any():
            self.df = self.df.groupby(self.df.index).mean()
            self.log_message("Обнаружены дубликаты в индексе. Агрегированы по среднему.")

        # Ресемплинг
        self.df = self.df.resample('H').interpolate(method='linear')
        self.log_message("Данные ресемплированы на почасовую частоту.")

        # Добавление циклических признаков
        self.df['day_of_year'] = self.df.index.dayofyear
        self.df['sin_day'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365.25)
        self.df['cos_day'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365.25)
        self.df = self.df.drop('day_of_year', axis=1)
        self.log_message("Добавлены циклические признаки дня года.")

        # Создание последовательностей
        X, y, timestamps = self.create_sequences(self.df)
        self.X, self.y, self.timestamps = X, y, timestamps

        # Масштабирование
        self.scale_data()

        self.status_label.config(text="Данные подготовлены. Готово к обучению.")
        self.log_message("Данные подготовлены. Всего последовательностей: " + str(len(self.X)))

    def create_sequences(self, df, lookback_wind=72, lookback_temp=6, forecast_horizon=6):
        X, y, timestamps = [], [], []
        wind_speed = df['wind_on_10meters'].values
        temp = df['temp'].values
        sin_day = df['sin_day'].values
        cos_day = df['cos_day'].values

        # Шаг в 6 часов для тестовой выборки, чтобы избежать наложения
        i = lookback_wind
        while i < len(df) - forecast_horizon:
            wind_seq = wind_speed[i-lookback_wind:i]
            temp_seq = np.zeros(lookback_wind)
            temp_seq[-lookback_temp:] = temp[i-lookback_temp:i]
            cyclic_features = [sin_day[i], cos_day[i]]
            seq = np.stack([wind_seq, temp_seq], axis=-1)
            X.append([seq, cyclic_features])
            y.append(wind_speed[i+1:i+forecast_horizon+1])
            timestamps.append(df.index[i])
            i += forecast_horizon if i >= len(df) * 0.85 else 1  # Шаг 6 часов только для тестовой выборки

        return X, np.array(y), timestamps

    def scale_data(self):
        wind_scaler = MinMaxScaler()
        temp_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        # Разделение на обучающую, валидационную и тестовую выборки
        train_size = int(0.7 * len(self.X))
        val_size = int(0.15 * len(self.X))
        self.X_train = self.X[:train_size]
        self.X_val = self.X[train_size:train_size + val_size]
        self.X_test = self.X[train_size + val_size:]
        self.y_train = self.y[:train_size]
        self.y_val = self.y[train_size:train_size + val_size]
        self.y_test = self.y[train_size + val_size:]
        self.test_timestamps = self.timestamps[train_size + val_size:]

        X_train_seq = [x[0] for x in self.X_train]
        X_train_wind = np.array([x[:, 0] for x in X_train_seq])
        X_train_temp = np.array([x[:, 1][-6:] for x in X_train_seq])
        wind_scaler.fit(X_train_wind)
        temp_scaler.fit(X_train_temp)
        target_scaler.fit(self.y_train.reshape(-1, 1))

        X_scaled = []
        for seq, cyclic in self.X:
            seq[:, 0] = wind_scaler.transform(seq[:, 0].reshape(1, -1)).flatten()
            seq[-6:, 1] = temp_scaler.transform(seq[-6:, 1].reshape(1, -1)).flatten()
            X_scaled.append([seq, cyclic])

        y_flat = self.y.reshape(-1, 1)
        y_scaled_flat = target_scaler.transform(y_flat)
        y_scaled = y_scaled_flat.reshape(-1, 6)

        self.X_scaled = X_scaled
        self.y_scaled = y_scaled
        self.scalers = {'wind': wind_scaler, 'temp': temp_scaler, 'target': target_scaler}

        self.X_train_seq = np.array([x[0] for x in self.X_train])
        self.X_train_cyclic = np.array([x[1] for x in self.X_train])
        self.X_val_seq = np.array([x[0] for x in self.X_val])
        self.X_val_cyclic = np.array([x[1] for x in self.X_val])
        self.X_test_seq = np.array([x[0] for x in self.X_test])
        self.X_test_cyclic = np.array([x[1] for x in self.X_test])
        self.y_train_scaled = self.y_scaled[:train_size]
        self.y_val_scaled = self.y_scaled[train_size:train_size + val_size]
        self.y_test_scaled = self.y_scaled[train_size + val_size:]

    def train_model(self):
        sequence_input = Input(shape=(72, 2), name='sequence_input')
        cyclic_input = Input(shape=(2,), name='cyclic_input')
        x = LSTM(64, return_sequences=True)(sequence_input)
        x = LSTM(32)(x)
        x = Concatenate()([x, cyclic_input])
        x = Dense(16, activation='relu')(x)
        output = Dense(6, activation='linear')(x)

        self.model = Model(inputs=[sequence_input, cyclic_input], outputs=output)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Обучение с выводом прогресса
        epochs = 20
        for epoch in range(epochs):
            history = self.model.fit(
                [self.X_train_seq, self.X_train_cyclic], self.y_train_scaled,
                validation_data=([self.X_val_seq, self.X_val_cyclic], self.y_val_scaled),
                epochs=1, batch_size=32, verbose=0
            )
            train_loss = history.history['loss'][0]
            val_loss = history.history['val_loss'][0]
            train_mae = history.history['mae'][0]
            val_mae = history.history['val_mae'][0]
            self.log_message(
                f"Эпоха {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}"
            )
        self.history = self.model.fit(
            [self.X_train_seq, self.X_train_cyclic], self.y_train_scaled,
            validation_data=([self.X_val_seq, self.X_val_cyclic], self.y_val_scaled),
            epochs=epochs, batch_size=32, verbose=0
        )

    def run_forecast(self):
        self.status_label.config(text="Обучение модели...")
        self.log_message("Начало обучения модели...")
        self.root.update()

        # Обучение модели
        self.train_model()

        # Прогноз на тестовой выборке
        y_pred_scaled = self.model.predict([self.X_test_seq, self.X_test_cyclic])
        y_pred_flat = y_pred_scaled.reshape(-1, 1)
        y_pred_unscaled = self.scalers['target'].inverse_transform(y_pred_flat).reshape(-1, 6)
        y_test_unscaled = self.scalers['target'].inverse_transform(self.y_test_scaled.reshape(-1, 1)).reshape(-1, 6)

        # Создание непрерывной временной серии для тестовой выборки
        actual_series = []
        predicted_series = []
        plot_timestamps = []
        for i in range(len(y_test_unscaled)):
            start_time = self.test_timestamps[i]
            for j in range(6):
                time_step = start_time + timedelta(hours=j+1)
                actual_series.append(y_test_unscaled[i, j])
                predicted_series.append(y_pred_unscaled[i, j])
                plot_timestamps.append(time_step)

        # Построение графика прогноза на тестовой выборке (сверху)
        self.ax1.clear()
        self.ax1.plot(plot_timestamps, actual_series, label='Актуальная', color='blue', marker='o', markersize=3)
        self.ax1.plot(plot_timestamps, predicted_series, label='Прогнозируемая', color='orange', marker='x', markersize=3)
        self.ax1.set_title('Прогноз на тестовой выборке (каждые 6 часов)')
        self.ax1.set_xlabel('Время')
        self.ax1.set_ylabel('Скорость ветра (м/с)')
        self.ax1.legend()
        self.ax1.grid(True)
        self.ax1.tick_params(axis='x', rotation=45)

        # Прогноз на последней последовательности (снизу)
        X_seq = np.array([self.X_scaled[-1][0]])
        X_cyclic = np.array([self.X_scaled[-1][1]])
        y_pred_scaled = self.model.predict([X_seq, X_cyclic])
        y_pred_flat = y_pred_scaled.reshape(-1, 1)
        y_pred = self.scalers['target'].inverse_transform(y_pred_flat).reshape(-1, 6)[0]

        self.ax2.clear()
        self.ax2.plot(range(1, 7), y_pred, label='Прогноз', marker='x', color='orange')
        self.ax2.set_title('Прогноз скорости ветра на 6 часов')
        self.ax2.set_xlabel('Часы вперед')
        self.ax2.set_ylabel('Скорость ветра (м/с)')
        self.ax2.legend()
        self.ax2.grid(True)

        # Установка расстояния между графиками
        self.fig.tight_layout(pad=3.0)

        self.canvas.draw()

        # Вычисление достоверности на тестовой выборке
        # 1. Вычисляем MAE для каждого шага (t+1 до t+6)
        mae_per_step = np.mean(np.abs(y_pred_unscaled - y_test_unscaled), axis=0)
        # 2. Вычисляем среднюю прогнозируемую скорость для каждого шага
        mean_pred_per_step = np.mean(y_pred_unscaled, axis=0)
        # 3. Вычисляем относительную ошибку (MAE / средняя прогнозируемая скорость)
        # Добавляем небольшую константу в знаменатель, чтобы избежать деления на 0
        relative_error_per_step = mae_per_step / (mean_pred_per_step + 1e-10)
        # 4. Вычисляем достоверность как 1 - относительная ошибка (в процентах)
        accuracy_per_step = (1 - relative_error_per_step) * 100
        # Ограничиваем достоверность диапазоном [0, 100]
        accuracy_per_step = np.clip(accuracy_per_step, 0, 100)

        accuracy_text = "Достоверность прогноза (%):\n" + "\n".join(
            f"t+{i+1}: {acc:.2f}%" for i, acc in enumerate(accuracy_per_step)
        )
        self.accuracy_label.config(text=accuracy_text)
        self.log_message("Достоверность прогноза на тестовой выборке:\n" + accuracy_text)

        self.status_label.config(text="Прогноз завершен!")
        self.log_message("Прогноз завершен.")

if __name__ == "__main__":
    root = tk.Tk()
    app = WindForecastApp(root)
    root.mainloop()