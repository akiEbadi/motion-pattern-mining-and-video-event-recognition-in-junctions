# junction event recognition (qmul junction) - deep learning version

this is a python re-do of the method in the MSc thesis _"Motion Pattern Mining and Video Event Recognition in Junctions"_ (chapter 4), but with deep learning instead of HDP / HMM-LDA / SVM.
input is the qmul junction video, output is a traffic state for every 75-frame clip.

persian notes are at the bottom (توضیحات فارسی پایین فایل هست)

---

## setup

python 3.10+

```
pip install -r requirements.txt
```

put the video in `data/videos/` (e.g. `qmul_junction.avi`). the video is NOT in the zip, too big.

## how to run

```
# 1) video -> optical flow per grid cell   (~16 min on 2 cores, uses all cores)
python main.py extract --video data/videos/qmul_junction.avi --out outputs/flow.npz

# 2a) unsupervised version (closest to the thesis): activities -> states -> test clips
python main.py run --words outputs/flow.npz --out outputs/run --labels labels/labels.csv

# 2b) supervised version (best accuracy, needs labels for the training clips)
python main.py classify --words outputs/flow.npz --labels labels/labels.csv --out outputs/classify

# 3) look at random test clips per class (pictures with arrows)
python main.py review --video data/videos/qmul_junction.avi --predictions outputs/classify/predictions.csv --out outputs/review
```

`--labels` is optional for `run` (without it you only get the discovered classes, no accuracy).
`outputs/flow.npz` is already in the zip so you can skip step 1.

first 500 clips = train, the other 700 = test (same as the thesis).

## classes

- `vertical` - main road traffic (north-bound on the left side, south-bound on the right, which shows up as SE in this camera)
- `left` - horizontal flow to the left (west)
- `right` - horizontal flow to the right (east)
- `turning` - most of the moving cars are turning from the main road into the side road (left/right turns, u-turns count too). if most cars go straight up/down it's `vertical` even when 1-2 cars turn ("majority" rule)
- `mixed` - vertical flow and turning happen together and neither is clearly the majority (about 40-65% of the motion is turning). these clips are real, not noise: the model picks them out at 0.80 f1
- `empty` - nothing moves through the junction (cars waiting at the red light don't count, they're not moving)

the first 4 are the thesis states. `empty` and `mixed` are new.
rule: if only one car moves, that car's direction decides the class (not empty).

## what's where

```
main.py              commands (extract / run / classify / review / all)
config.py            all settings
src/features.py      step 1: flow, direction words, bag of words, empty-junction test
src/activities.py    step 2: neural topic model (instead of HDP) + thesis eqs 3-9
src/states.py        step 3: hybrid neural HMM (instead of HMM-LDA), online filter
src/kmeans.py        k-means (thesis approach 3), used as baseline
src/classifier.py    step 4 supervised: ensemble of 2 small neural nets (instead of SVM)
src/evaluation.py    naming classes, hungarian matching, accuracy, plots
src/review.py        picture sheets for checking by eye
labels/              the labels + how they were made
outputs/             results of the last run
```

## thesis vs this code

| step            | thesis                                               | here                                                                                                     |
| --------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1 clips + words | 75 frames, farneback flow, 10x10 cells, 8 directions | same. directions are centred on E, SE, S... (an early version had the bins shifted by half a bin, fixed) |
| 2 activities    | HDP + eqs 3-8                                        | neural topic model (VAE) + same eqs 3-8                                                                  |
| 2 clip vector   | eq 9                                                 | same                                                                                                     |
| 3 states        | HMM-LDA                                              | hybrid neural HMM (MLP emissions + learned transitions, EM)                                              |
| silent clips    | eq 14 (threshold 0.08)                               | replaced by an `empty` class (eq 9 is normalised per clip so it can't see empty clips)                   |
| 4 test clips    | SVM / k-means                                        | `classify`: 2 neural nets averaged (bag of words net + activity vector net)                              |

not done: the GP classifier.
tried and dropped: the key-activity rule for turning (thesis eq 15), it didn't help here (91.1% vs 92.1% without it).

## results (700 test clips)

accuracy against `labels/labels.csv`:

| method                                                                        | acc       |
| ----------------------------------------------------------------------------- | --------- |
| hybrid neural HMM (`run`)                                                     | 86.0%     |
| k-means on activity vectors (thesis approach 3, baseline in `run`)            | 92.1%     |
| k-means + key-activity rule                                                   | 91.1%     |
| CNN on cell-direction maps                                                    | ~92-93%   |
| ensemble of 2 neural nets (`classify`), old labels                            | 94.0%     |
| ensemble of 2 neural nets (`classify`), 4 classes, majority rule              | 94.1%     |
| **ensemble of 2 neural nets (`classify`), 5 classes incl. `mixed` (current)** | **94.1%** |

per class with the current labels: right 1.00 f1, vertical 0.98, left 0.94, empty 0.89, mixed 0.80, turning 0.64.
on the test clips labelled by a person (you or me by eye) it's ~77% - those are the hard ones, not a random sample.
adding `mixed` cost nothing overall (94.1% both ways) and made `vertical` cleaner (0.97 -> 0.98). it moved errors out of
vertical and into the smaller turning class, which is the honest place for them.

### things tried that did NOT help

- **reach / extent features** (see `reach_features` in src/features.py). the real difference between a left turn and
  westward flow is not direction but how far the motion reaches: westward motion sits in the right half of the frame
  44% of the time for `left` clips but only 14% for `turning` clips. clear in the statistics, useless in the model:
  the bag of words already holds the same thing per cell, and `turning` has only ~34 training clips. a threshold rule
  on it scores 87.8% on left-vs-turning while "always say left" scores 89.1%, because the classes are that imbalanced.
- region x direction maps, CNN on time-segmented direction maps, previous-clip context, causal HMM smoothing over
  the clip sequence, class re-weighting tuned by cross-validation. all within noise of 94%.

**careful with these numbers:**

- most labels (~95%) come from a rule over the same optical flow the models use, so the comparison is a bit circular.
- a blind check by eye (82 test clips, judged without seeing label or prediction) puts the real accuracy of `classify` at about **93.7%**, and the labels themselves at about 95.5%.
- the thesis numbers (99.15% / 95.62%) are from the thesis' own labelled experiments, NOT from this code.
- the 40 test clips labelled by a person are the hard ones, accuracy there is only ~50%. not a random sample.

## labels

`labels/labels.csv` = what the code reads (clip_id,label).
`labels/labels_for_review.csv` = same + where each label came from (`source`: you / visual check / rule) and why a clip is doubtful.

how they were made:

1. rule over each clip's direction mix (see `labels/auto_label_rules.py`, it uses absolute paths from my machine, fix them if you run it)
2. a few hundred clips checked by eye, wrong ones fixed
3. hand labels (`labels/your_corrections.csv`) override everything

known problems:

- turning definition is majority (see classes). all vertical/turning clips with real turning motion (~230) were re-checked by eye with that rule.
- the half-and-half clips are now their own class (`mixed`), so they no longer need a forced yes/no answer.
- some hand labels were given before the majority rule ("turning + vertical", "turning + left") and still say turning.
  those need a re-check with the new definition: `labels/recheck_clips.pdf` has them. most remaining errors are there.
- the left-vs-turning confusion (13 clips) is the biggest single error group and it is a small-class problem, not a
  missing-feature one. more labelled turning clips would help more than any feature.
- clips 702 and 1042 have two cars going different ways, still doubtful.
- `labels/blind_audit.csv` = the blind check results.

## other notes

- the dataset page says 25 Hz, the file header says 29.97 fps. clip = 75 frames anyway (3 s or 2.5 s).
- directions are image directions: N = up in the frame, S = down, E = right, W = left.
- the empty test (config.py, `EMPTY_*`) is tuned for this camera and the 10x10 grid (45/46 right on clips checked by eye). another camera = retune it.
- clip 264 has an ambulance going against traffic. that's an anomaly, not really one of these classes. the dataset has a separate anomaly ground truth (`accv_2010_dataset.zip` on the dataset page).
- K=6 states in `run` was worse than 4 (64%).

## todo

- [x] decide the turning definition, re-check vertical/turning labels, retrain
- [x] add a `mixed` class for the half-and-half clips
- [ ] re-check the old hand labels (labels/recheck_clips.pdf) with the majority rule
- [ ] more labelled `turning` clips - that class has ~34 training examples and carries most of the error
- [ ] fresh blind check of random test clips after that
- [ ] maybe a CNN on the raw flow over time once the labels are clean (didn't beat the MLPs with the current labels)

---

## توضیحات فارسی

### این پروژه چیست

این کد روش فصل ۴ پایان‌نامه را با پایتون و روش‌های deep learning پیاده می‌کند. ویدیوی تقاطع QMUL به کلیپ‌های ۷۵ فریمی تقسیم می‌شود و برای هر کلیپ وضعیت ترافیکی مشخص می‌شود. ۵۰۰ کلیپ اول برای آموزش و ۷۰۰ کلیپ بعدی برای تست است، همان‌طور که در پایان‌نامه بود.

### مراحل

1. **استخراج ویژگی:** جریان نوری بین فریم‌ها محاسبه می‌شود. هر فریم به ۱۰×۱۰ سلول تقسیم می‌شود و جهت حرکت هر سلول به یکی از ۸ جهت تبدیل می‌شود. هر (سلول، جهت) یک «کلمه» است.
2. **کشف فعالیت‌ها:** به‌جای HDP یک مدل موضوعی عصبی (VAE) فعالیت‌ها را یاد می‌گیرد. روابط ۳ تا ۹ پایان‌نامه برای انتخاب فعالیت‌ها و ساختن بردار هر کلیپ بدون تغییر استفاده شده‌اند.
3. **کشف وضعیت‌ها (بدون برچسب):** به‌جای HMM-LDA یک HMM ترکیبی عصبی استفاده شده است. دستور `run` این مرحله را اجرا می‌کند.
4. **طبقه‌بندی داده‌ی تست (با برچسب):** دو شبکه‌ی عصبی کوچک روی کلیپ‌های آموزشی برچسب‌دار آموزش می‌بینند و احتمال‌هایشان میانگین گرفته می‌شود. دستور `classify` این مرحله را اجرا می‌کند و بهترین دقت را دارد.

### کلاس‌ها

- **vertical:** جریان خیابان اصلی (شمال‌رو سمت چپ، جنوب‌رو سمت راست)
- **left:** جریان افقی به سمت چپ (غرب)
- **right:** جریان افقی به سمت راست (شرق)
- **turning:** بیشتر ماشین‌های در حال حرکت از خیابان اصلی به چپ یا راست می‌پیچند (دور زدن هم حساب است). اگر بیشتر ماشین‌ها مستقیم بالا یا پایین بروند، کلیپ vertical است، حتی اگر یکی دو ماشین بپیچند.
- **mixed:** جریان عمودی و گردش با هم اتفاق می‌افتند و هیچ‌کدام اکثریت روشنی ندارند.
- **empty:** هیچ خودرویی از تقاطع عبور نمی‌کند. ماشین‌هایی که پشت چراغ قرمز ایستاده‌اند حساب نمی‌شوند.

اگر فقط یک ماشین در حال حرکت باشد، جهت همان ماشین کلاس را تعیین می‌کند.

### نتیجه

دقت روی ۷۰۰ کلیپ تست **۹۴٫۱٪** است (دستور `classify`). K-means به ۹۲٫۱٪ و HMM عصبی به ۸۶٪ رسید.

اضافه کردن کلاس mixed هیچ هزینه‌ای نداشت: دقت با ۴ کلاس و با ۵ کلاس هر دو ۹۴٫۱٪ است، ولی کلاس vertical تمیزتر شد (۰٫۹۷ به ۰٫۹۸) و برچسب‌ها صادقانه‌تر شدند.

روی کلیپ‌های تستی که با چشم برچسب خورده‌اند دقت حدود ۷۷٪ است. این‌ها سخت‌ترین کلیپ‌ها هستند و نمونه‌ی تصادفی نیستند.

### تفکیک «گردش به چپ» از «جریان افقی چپ»

این مشکل را بررسی کردم و ویژگی درست را پیدا کردم: **تفاوت در جهت نیست، در بُرد حرکت است.** ماشینی که در خیابان افقی می‌راند کل عرض قاب را طی می‌کند، ولی ماشینی که می‌پیچد فقط تا وسط قاب می‌آید. در کلیپ‌های left حدود ۴۴٪ حرکت غربی در نیمه‌ی راست قاب است، ولی در کلیپ‌های turning فقط ۱۴٪.

ولی این ویژگی دقت مدل را بالا نبرد و تابع `reach_features` در `src/features.py` با همین توضیح نگه داشته شده است. دو دلیل دارد:
۱. همین اطلاعات از قبل در ویژگی‌های سلول×جهت وجود دارد، چون آن‌ها هم مکان را نشان می‌دهند.
۲. کلاس turning فقط حدود ۳۴ نمونه‌ی آموزشی دارد. از ۲۲۱ کلیپی که حرکت غربی واقعی دارند، ۱۹۷ تا left و ۲۴ تا turning است. بهترین قاعده روی این ویژگی ۸۷٫۸٪ درست عمل می‌کند، در حالی که «همیشه بگو left» به تنهایی ۸۹٫۱٪ می‌دهد.

یعنی مشکل کمبود ویژگی نیست، کمبود نمونه است. برای بهتر شدن، نمونه‌ی آموزشی بیشتر از کلاس turning لازم است، نه ویژگی جدید.

### مشکل باقی‌مانده

بزرگ‌ترین گروه خطا ۱۳ کلیپ turning است که مدل left می‌خواند. چند برچسب دستی هم قبل از تعریف اکثریت داده شده‌اند (مثل «turning + vertical») و هنوز turning هستند. فایل `labels/recheck_clips.pdf` این‌ها را دارد.

### فایل‌های مهم خروجی

- `outputs/classify/predictions.csv`: کلاس پیش‌بینی‌شده و احتمال هر کلاس برای همه‌ی کلیپ‌ها
- `outputs/classify/metrics.txt` و `confusion_matrix.png`: دقت روی داده‌ی تست
- `outputs/review/`: تصویر نمونه‌کلیپ‌های هر کلاس برای بررسی چشمی
- `labels/labels_for_review.csv`: برچسب همه‌ی کلیپ‌ها و منبع هر برچسب
