"""
Калибровка камеры по шахматной доске (chessboard).

Зачем: чтобы получить реальные intrinsics камеры (FX, FY, CX, CY) и коэффициенты
дисторсии вместо примерных значений в phone_detect.py.

Подготовка:
  1. Распечатайте шахматную доску (например, 9x6 внутренних углов, стандартный
     калибровочный паттерн OpenCV: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png).
  2. Измерьте реальный размер одной клетки доски в метрах (SQUARE_SIZE_M).
  3. Приклейте доску на что-то жёсткое и ровное (иначе калибровка будет неточной).

Использование:
  python calibrate_camera.py
  - 's' — сохранить текущий кадр как калибровочный (доска должна быть найдена, углы подсвечены зелёным)
  - 'c' — запустить калибровку по накопленным кадрам (нужно минимум 10-15 кадров)
  - 'q' — выйти без калибровки

Снимайте доску под разными углами и с разных расстояний, включая края кадра —
это критично для точной оценки дисторсии.

Результат сохраняется в camera_calibration.json и выводится в виде готовых
строк FX/FY/CX/CY для вставки в phone_detect.py.
"""

import json

import cv2
import numpy as np

# Количество внутренних углов доски (по горизонтали, по вертикали).
# Для стандартной доски 10x7 клеток это будет (9, 6).
CHESSBOARD_SIZE = (9, 6)

# Реальный размер одной клетки доски, в метрах
SQUARE_SIZE_M = 0.024

MIN_FRAMES_FOR_CALIBRATION = 10


def build_object_points() -> np.ndarray:
    cols, rows = CHESSBOARD_SIZE
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M
    return objp


def main():
    objp = build_object_points()

    object_points = []  # 3D точки в пространстве доски, для каждого кадра
    image_points = []  # соответствующие 2D точки на изображении

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("Наведите камеру на шахматную доску под разными углами.")
    print("'s' — сохранить кадр, 'c' — калибровать, 'q' — выйти")

    image_size = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_size = frame.shape[1::-1]  # (width, height)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        display = frame.copy()
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)

        status = f"Сохранено кадров: {len(object_points)}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            if found:
                object_points.append(objp)
                image_points.append(corners)
                print(f"Кадр сохранён ({len(object_points)} всего)")
            else:
                print("Доска не найдена на кадре — не сохранено")
        elif key == ord("c"):
            if len(object_points) < MIN_FRAMES_FOR_CALIBRATION:
                print(f"Нужно минимум {MIN_FRAMES_FOR_CALIBRATION} кадров, сейчас {len(object_points)}")
                continue
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(object_points) < MIN_FRAMES_FOR_CALIBRATION:
        print("Недостаточно кадров для калибровки, выход без сохранения результата.")
        return

    print("Калибровка...")
    rms_error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    result = {
        "rms_reprojection_error": float(rms_error),
        "image_size": image_size,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "dist_coeffs": dist_coeffs.flatten().tolist(),
    }

    with open("my_iphone_camera_calibration.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nRMS ошибка репроекции: {rms_error:.4f} (хорошо, если < 0.5-1.0 px)")
    print("Результат сохранён в camera_calibration.json")
    print("\nВставьте в phone_detect.py:")
    print(f"FX = {fx:.2f}")
    print(f"FY = {fy:.2f}")
    print(f"CX = {cx:.2f}")
    print(f"CY = {cy:.2f}")


if __name__ == "__main__":
    main()
