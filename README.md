# Итоги

## Кратко

На данных CIFAR-10 обучил автоэнкодер и классификатор.

Содержание:
1. Автоэнкодер
2. Классификатор
3. Выводы
4. Технические детали

![Исходная задача](Тестовое%20задание.md)

## Автоэнкодер
Для автоэнкодера выбрал простейшую архитектуру: две свертки в одну сторону и две в обратную.

![autoencoder_model](resources/autoencoder_model.png)
  
### Выбор архитектуры
При выборе учитывал следующие моменты:
* Скрытое представление должно быть размерности ниже, чем исходное → $x < 32 * 32 * 3 = 3072$
* Скрытое представление не должно быть слишком маленьким — у нас 10 классифицируемых категорий, и внутри каждой объекты могут заметно отличаться по форме и цвету
* Свертка лучше сохраняет пространственную информацию для изображений
* Модель не должна быть слишком большой, чтобы модель показала хоть какой-то результат на моем слабеньком ноутбуке за разумное время
* Плохое качество модели обещали не оценивать :)

### Качество

Качество автоэнкодера решил оценивать по функции потерь (MSE) плюс визуальный контроль. 

В реальной задаче можно было бы посмотреть на специализированные метрики качества из области сжатия изображений. Например, structural similarity (ssim) или psnr. Но здесь это кажется лишним.

С train/val/test сплитом тоже не стал заморачиваться — ради экономии времени применю это только на этапе классификатора.

Насколько можно судить по ошибке на train и пикселированной визуализации, то сами изображения восстанавливаются неплохо. Однако на некоторых из них проявляются явно заметные цветовые артефакты. Из свертки, вероятно? Stride, наверно, надо было оставить единицей. Надо бы разобраться... а может и так сработает? Рискну :)

Batch MSE (log scale)
![autoencoder_quality](resources/1_autoencoder_quality.png)
Сами картинки
![autoencoder_quality](resources/2_autoencoder_quality.png)


## Классификатор

Реализовал простейшую нейронную сеть из нескольких полносвязных слоев.

### Пайплайн

Тестовый датасет разбил на три части: 

Для тестирования модели данные разбил на три части — train/val/test (48K/2K/10K изображений). Тюнинг делал на train/val, финальный замер качества делал на test.

О размерах датасетов. Размер тестового был задан из коробки, а для валидации случайным образом вынул из train 2 тысячи картинок, больше не нужно.

### Качество  классификатора

Классы в датасете сбалансированы, а это значит, что можно без проблем смотреть на долю правильных ответов (Accuracy). Дополнительно я еще посматривал на матрицу ошибок, она довольно наглядная.


Теперь к цифрам. Текущий расклад по качеству такой: https://paperswithcode.com/sota/image-classification-on-cifar-10
![clf_sota](resources/clf_sota.png)

Лучший результат в мире — 99.5%. 
В этом тестовом задании, наверное, реально получить 85-95%.
Моя нетюнингованная модель: 47%
![classifier_quality](resources/classifier_quality.png)

## Выводы (ака следующие шаги)

Модель плохая, что же делать? По-хорошему, надо проводить анализиз ошибок на train/val/test датасетах. Там выясниться, в какую сторону стоит дальше копать.

Скорее всего, помочь может что-то из списка ниже.

#### Улучшить автоенкодер

* Попробовать другие архитектуры
	* fully-connected layers
	* batchnorm
	* maxpooling
- Нормализовать входные данные
	- (почему-то во всех туториалах cifar10 нормализуют неправильно)
- Попробовать регуляризацию
- Попробовать аугументацию датасета
- Обучать дольше

В принципе, подготовленная инфраструктура позволяет это все достаточно просто сделать.

#### Улучшить классификатор
- На удачу, не вникая и не тратя много особо времени
	- Попробовать несколько fully-connected layers
	- Попробовать выдавать из автоэнкодера unflatten представление
- Посмотреть, что делают топовые модели
- Ну и начать проводить анализ ошибок на train/val/test датасете (чтобы улучшать выбраную модель)
- И здесь аугументацию датасета попробовать
- Обучать дольше тоже можно

### О демонстрации

Формат демонстрации зависит от того, кто целевая аудитория. Клиенту нужно одно, коллегам второе, журналу для публикации — третье. В целом, Tensorboard себя неплохо зарекомендовал и я бы делал его отправной точной для всего остального.


## Технические детали

### Установка
```
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Структура репозитория

* data → данные для обучения

* models → веса для моделей

* notebooks → тетрадки с экспериментами

	* notebooks/runs → сырые логи экспериментов (обычно в .gitignore, но это тестовое)

* resources → картинки для readme.md

* src → все рутины для обучения, код финализированных моделей и прочие полезные инструменты

* tests → тесты для контроля и дебага вынесенных из тетрадок модулей

* setup.py → базовый скрипт, чтобы сделать все пакеты доступными из любой папки + задел на упаковку сервис

### Рабочая среда

* Автоэнкодер писал на PyTorch
* Классификатор на PyTorch Lightning
* Для визуализации использовал TensorBoard
* Пару тестов написал на PyTest  
