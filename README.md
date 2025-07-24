# head_detection_cpu
Human Head Detection with YOLOv8 on CPU


## Human Head Detection with YOLOv8 on CPU

این پروژه با هدف شناسایی سر انسان در ویدیوبا استفاده از YOLOv8 طراحی شده است و به‌صورت کاملاً بهینه برای سیستم‌های بدون GPU(فقط CPU) پیاده‌سازی شده است.

---

## ویژگی‌ها

- تشخیص سر انسان در ویدیو با bounding box
- نمایش و ثبت میانگین FPS واقعی
- ذخیره ویدیوی خروجی با جعبه‌های تشخیص
- بهینه‌سازی برای اجرا روی CPU

---

## پیش‌نیازها

ابتدا مطمئن شوید که Python 3.11 نصب شده باشد.

### نصب کتابخانه‌ها (داخل محیط مجازی)

```bash
python -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install ultralytics opencv-python

```

----------

## 🛠️ فایل‌ها

-   `head_detection_cpu.py` — کد اصلی پروژه
    
-   `nano.pt` — فایل مدل YOLO سبک و سریع
    
-   `input.mp4` — ویدیوی تستی (دلخواه)
    
-   `output.mp4` — ویدیوی خروجی با bounding boxها
    

----------

## 🚀 نحوه اجرا

1.  فایل وزن `nano.pt` را از [مدل‌های YOLOv8](https://github.com/Abcfsa/YOLOv8_head_detector?utm_source=chatgpt.com
) دانلود کرده و در پوشه پروژه قرار دهید.
    
2.  فایل ویدیوی تستی خود را در پوشه بگذارید (`input.mp4`)
    
3.  سپس با اجرای دستور زیر محیط مجازی رافعال کنید:
    

```bash
.\venv\Scripts\activate
```
4. پس از فعال شدن محیط مجازی با کد زیر برنامه را اجرا کنید:

```bash
python head_detection_cpu.py
```
----------

## 📈 خروجی

-   ویدیوی خروجی: `output_video.mp4`
    
-   میانگین FPS واقعی در پایان چاپ می‌شود:
    

