import cv2

# 加载图像
for i in range(52, 57):
    image = cv2.imread("{}.jpg".format(str(i)))

    # 检查图像是否成功加载
    if image is not None:
        # 获取图像的宽度和高度
        height, width = image.shape[:2]

        # top = height // 2  # 下半脸的顶部位置为图像高度的一半
        # bottom = height - 1  # 下半脸的底部位置为图像高度减1
        # left = 0  # 下半脸的左侧位置为图像的左侧
        # right = width - 1  # 下半脸的右侧位置为图像的右侧
        #
        # image[top:bottom, left:right] = 0.
        # 定义红色边框的厚度
        thickness = 7
        # 定义红色的边框颜色 (BGR格式)
        red_color = (0, 0, 255)

        # 在图像上绘制红色边框
        image_with_border = cv2.rectangle(image, (0, 0), (width - 1, height - 1), red_color, thickness)

        # 显示带有红色边框的图像
        cv2.imwrite("{}.jpg".format(str(i)), image_with_border)
    else:
        print("Failed to load image.")