# CNN-Comparison

So sánh các biến thể **CNN** trên FashionMNIST: kiến trúc nhiều lớp dropout và
các hàm kích hoạt khác nhau.

## Nội dung

- `models/CNN.py` — CNN cơ bản.
- `models/CNN2.py` — `CNN_Dropout3`: nhiều lớp conv + dropout.
- `models/ResNet.py` — tham chiếu ResNet.
- `train_cnn.py` — lớp `Trainer`: huấn luyện, đánh giá, lưu kết quả.
- `main.py` — chạy thí nghiệm với lần lượt `ReLU`, `SiLU`, `LeakyReLU`, `GELU`
  (40 epoch mỗi cấu hình), lưu biểu đồ và checkpoint vào `results/`.

## Chạy

```bash
pip install -r requirements.txt
python main.py
```

Kết quả mỗi lần chạy nằm trong `results/<model>/<activation>/<timestamp>/`.
