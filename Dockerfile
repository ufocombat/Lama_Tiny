# Используем официальный образ Python версии 3.11 как базовый
FROM python:3.11

# Устанавливаем рабочую директорию в контейнере
WORKDIR /usr/src/app

# Копируем файл зависимостей в текущую директорию
COPY requirements.txt ./

# Устанавливаем зависимости
RUN pip3 install --no-cache-dir --default-timeout=100 -r requirements.txt

# Устанавливаем Uvicorn для запуска нашего приложения
RUN pip install uvicorn

# Устанавливаем transformers напрямую из репозитория на GitHub
RUN pip install git+https://github.com/huggingface/transformers.git

# Копируем все файлы из текущей директории в контейнер
COPY . .

# Задаем команду для запуска приложения через Uvicorn с автоперезагрузкой
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--reload"]
