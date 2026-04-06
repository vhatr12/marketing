# Customer Marketing Dataset — EDA & Linear Regression

## Mô tả project

Phân tích dataset marketing khách hàng (2,240 quan sát, 24 biến) nhằm:
- Khám phá đặc điểm hành vi và chi tiêu của khách hàng
- Xây dựng mô hình Linear Regression dự đoán **tổng chi tiêu** (`TotalSpending`)
- Đánh giá mức độ ảnh hưởng của thu nhập, số con, trình độ học vấn và tình trạng hôn nhân

## Dataset

`Customer_marketing_dataset__1_.xlsx` — sheet `marketing_campaign`

| Nhóm biến | Các cột |
|---|---|
| Demographics | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Chi tiêu | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts` |
| Hành vi mua | `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumDealsPurchases` |
| Campaign | `AcceptedCmp1/2/3`, `Response` |

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy notebook

```bash
jupyter notebook customer_marketing_regression.ipynb
```

Hoặc mở bằng VS Code với extension **Jupyter**.

## Cấu trúc notebook

| Section | Nội dung |
|---|---|
| 0 | Import libraries |
| 1 | Load dataset |
| 2 | Data cleaning & feature engineering |
| 3 | Exploratory Data Analysis (EDA) |
| 4 | Preprocessing |
| 5 | So sánh 4 mô hình Linear Regression |
| 6 | Bảng tổng hợp R² và ΔR² |
| 7 | Đánh giá mô hình tốt nhất (Actual vs Predicted, Residuals) |
| 8 | Kết luận |

## Kết quả chính

- **Income** là yếu tố quyết định nhất (R² = 0.629 chỉ với 1 biến)
- **Số con** có ảnh hưởng âm rõ rệt: không con chi $1,040 vs có 2 con chi $221
- **Education & Marital Status** không giải thích thêm đáng kể khi kiểm soát Income
- **Mô hình đầy đủ** (Model 4) đạt R² ≈ 0.79 trên test set
