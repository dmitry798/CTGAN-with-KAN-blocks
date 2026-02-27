# CTGAN-with-KAN-blocks

https://conference.kf.bmstu.ru/ - туда нам надо публиковаться

### Приблизительный план:
1) Нам нужно собрать пару датасетов по кредитному скорингу, проанализировать их
2) Использовать на них обычный CTGAN, потом модифицировать его используя KAN и снова посмотреть че вышло
3) Провести eval на метриках utility, fidelity, privacy, написать выводы: лучше или хуже KAN работает, чем обычный CTGAN

#### 1. Ссылки на датасеты:
1. https://www.kaggle.com/competitions/bank-scoring-case2/data - 11мб, 4 файла
2. https://www.kaggle.com/competitions/alfa-bank-pd-credit-history/data - 523мб, 18 файлов
3. https://www.kaggle.com/competitions/beeline-credit-scoring-competition-2/data - 8мб, 3 файла

#### 2. Модели:
1. https://github.com/sdv-dev/CTGAN
2. https://github.com/KindXiaoming/pykan/tree/master/kan

### Статьи:
1. "KAN может обучаться и предсказывать распределения точнее, чем нейросети (MLP), используя меньшее число параметров".
Они используют KAN как генератор и доказывают, что он лучше ловит сложные формы данных (fidelity), чем классические сети. Это ровно то, что мы увидели на твоем графике "двугорбого" распределения." - https://arxiv.org/abs/2512.11014
2. предлагает вставлять KAN в Дискриминатор GAN - https://onepetro.org/SJ/article-abstract/30/10/5932/787890/KA-GAN-Kolmogorov-Arnold-Generative-Adversarial?redirectedFrom=fulltext
3. Статья показывает использование KAN внутри WGAN (Wasserstein GAN) для предсказания свойств сплавов.
Результат: "AAKAN-WGAN лучше улавливает сложные нелинейные связи и генерирует данные высокой точности (high-fidelity)" - https://www.sciencedirect.com/science/article/abs/pii/S2352492825017106
4. Минусы: Скорость: KAN учится в 5-10 раз медленнее обычных сетей (из-за сложности сплайнов). Для генерации 100 строк таблицы это неважно, но для обучения на всем интернете — критично.
Память: Сплайны жрут больше памяти при обучении. - https://arxiv.org/html/2408.11200
5. https://github.com/JLZml/Credit-Scoring-Data-Sets - там еще какие то статьи
