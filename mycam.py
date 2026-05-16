from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO(r"F:/Code/Anaconda/yolov11/yolo26x-pose.pt")  # 或 yolo26n-pose.pt

# 打开摄像头进行实时推理
results = model(
    source=0,      # 0 表示默认摄像头
    stream=True,   # 流式处理模式
    device="gpu"   # 使用 CPU 进行推理 (如果有 GPU 可改为 "gpu"
)

for result in results:
    plotted_frame = result.plot()  # 绘制检测结果
    cv2.imshow("YOLO Inference", plotted_frame)  # 显示窗口（修正了变量名）
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
