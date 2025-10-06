# MARKOWIT_PPO_WEB
# THUẬT TOÁN PROXIMAL POLICY OPTIMIZATION TRONG TỐI ƯU HÓA DANH MỤC ĐẦU TƯ VÀ XÂY DỰNG CHIẾN LƯỢC ĐẦU TƯ CỔ PHIẾU
## Giới thiệu
Trong bối cảnh **thị trường chứng khoán Việt Nam** ngày càng mở rộng và biến động mạnh, nhu cầu về một **mô hình đầu tư khoa học, thích ứng và kiểm soát rủi ro** trở nên cấp thiết.  
Nghiên cứu này hướng đến việc **xây dựng và kiểm chứng mô hình kết hợp** giữa:

- **Lý thuyết danh mục đầu tư Markowitz (Modern Portfolio Theory – MPT)**
- **Thuật toán học tăng cường Proximal Policy Optimization (PPO)**
---
## Phương pháp
### 1. Giai đoạn Markowitz (MPT)
Chọn lọc **danh mục cổ phiếu tối ưu ban đầu trong rổ VN30**, dựa trên:
- Lợi suất kỳ vọng  
- Rủi ro (phương sai)  
- Hệ số Sharpe  
Kết quả là một **danh mục nền tảng cân bằng giữa rủi ro và lợi nhuận.**
---
### 2. Giai đoạn PPO (Reinforcement Learning)
Triển khai **mô hình PPO** trong **môi trường mô phỏng giao dịch tùy chỉnh** dựa trên OpenAI Gym.  
Sử dụng dữ liệu lịch sử gồm:
- Giá, khối lượng  
- Các chỉ báo kỹ thuật: **MA, RSI, MACD, Bollinger Bands**, v.v.  
PPO được huấn luyện để **ra quyết định động (mua – bán – giữ)** và **tối ưu hàm phần thưởng** gắn với:
- Tỷ suất lợi nhuận điều chỉnh rủi ro (**Sharpe Ratio**)  
- Độ sụt giảm tối đa (**Maximum Drawdown**)  
---
## Kết quả
Mô hình **Markowitz–PPO** cho thấy **hiệu quả vượt trội** so với các chiến lược truyền thống như:
- Momentum  
- Mean Reversion  
- Scalping  
Tăng đáng kể **Sharpe Ratio** và **hiệu suất tổng thể của danh mục.**
---
## Ý nghĩa & Ứng dụng
- Hỗ trợ **nhà đầu tư cá nhân** ra quyết định dựa trên dữ liệu và mô hình học máy.  
- Cung cấp **giải pháp tự động hóa phân bổ danh mục** cho **các quỹ đầu tư.**  
- Đề xuất **tham khảo cho cơ quan quản lý** trong việc phát triển **khung giám sát giao dịch thuật toán** tại Việt Nam.  
---
## Hạn chế & Hướng nghiên cứu tiếp theo
- **Độ phức tạp tính toán cao**, yêu cầu **dữ liệu chất lượng cao** và **tham số PPO nhạy cảm.**  
- **Tương lai có thể:**
  
  Tích hợp dữ liệu vĩ mô

  Mô hình hóa hành vi thị trường

  Cải thiện khả năng thích ứng của mô hình PPO
