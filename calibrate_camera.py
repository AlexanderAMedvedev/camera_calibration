"""
Калибровка камеры по шахматной доске.

Зачем:  
1) получить intrinsics камеры (f_x, f_y, c_x, c_y)
2) коэффициенты искажения 

Подготовка:
  1. Распечатайте шахматную доску 
  (например, 9x6 внутренних углов, стандартный
   калибровочный паттерн OpenCV: 
   https://github.com/opencv/opencv/blob/4.x/doc/pattern.png).
  2. Измерьте реальный размер одной клетки доски в метрах и 
  пропишите его в этом скрипте (см. переменную SQUARE_SIZE_M)
  3. Приклейте доску на что-то жёсткое и ровное

Снимайте доску 
1. под разными углами (включая края кадра для учёта искажений) 
2. с разных расстояний
"""

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# длина стороны одной клетки доски (метры)
SQUARE_SIZE_M = 0.024
# Количество внутренних углов доски (по горизонтали, по вертикали)
# Для доски 10 на 7 клеток это будет (9, 6).
CHESSBOARD_SIZE = (9, 6)
# Конфиг с источником видеопотока дб в папке рядом с папкой этого скрипта
# Формат конфига: {"video_source": "rtsp://user:pass@192.168.1.10:554/stream1"}
CONFIG_PATH = Path(__file__).resolve().parent.parent / \
'camera_config/camera_config.json'
# Eсли конфига нет: 0 — используем встроенную камеру
DEFAULT_VIDEO_SOURCE = 0
# Директория для кадров, по которым сделали калибровку
FRAMES_DIR = Path(__file__).resolve().parent/'frames'
MIN_FRAMES_FOR_CALIBRATION = 1
# Сколько подряд неудачных чтений кадра терпим, 
# прежде чем считать поток мёртвым.
# У сетевой камеры единичный сбой — это норма, а не конец потока: 
# переподключение TCP, ожидание ключевого кадра, потерянный пакет. 
# 30 попыток по 100 мс — около 3-x секунд: короткий обрыв переживём, 
# реально мёртвый поток не будет висеть.
MAX_READ_FAILURES = 30
READ_RETRY_DELAY_MS = 100

def load_video_source() -> int | str:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)['video_source']
    except (OSError, ValueError, KeyError) as error:
        print(f'Конфиг {CONFIG_PATH} не прочитан ({error}),' +
               'используем встроенную камеру')
        return DEFAULT_VIDEO_SOURCE


def open_video_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, str) and source.startswith('rtsp://'):
        # У VideoCapture нет параметра для транспорта RTSP, ffmpeg читает опции
        # из этой переменной окружения в момент открытия потока. TCP вместо UDP:
        # при потере пакетов кадр приходит с артефактами, а по нему ищут углы
        # доски с субпиксельной точностью — битый кадр испортит калибровку.
        os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS',
                              'rtsp_transport;tcp')

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError('Не удалось открыть источник видео')
    return cap


def save_calibration_frame(session_dir: Path, index: int, frame: np.ndarray, 
                           corners: np.ndarray) -> None:
    """
    PNG (а не JPEG): артефакты сжатия сдвигают углы при повторном поиске
    """
    session_dir.mkdir(parents=True, exist_ok=True)

    annotated = frame.copy()
    cv2.drawChessboardCorners(annotated, CHESSBOARD_SIZE, corners, True)

    for suffix, image in (('', frame), ('_corners', annotated)):
        path_name = session_dir / f'frame_{index:02d}{suffix}.png'
        if not cv2.imwrite(str(path_name), image):
            print(f'Не удалось записать {path_name}')


def build_object_points(chessboard_size: tuple[int, int],
                         square_size_m: float) -> np.ndarray:
    """
    точки с системе координат доски
    """
    cols, rows = chessboard_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


# 3D координаты углов доски в её системе координат (z = 0)
OBJECT_POINTS_3D = build_object_points(CHESSBOARD_SIZE, SQUARE_SIZE_M)


def main():
    print('START')
    object_points = []  # внутренние узлы доски в её системе координат
    image_points = []  # соответствующие 2D точки на фото-фрейме
# фрейм - отдельный кадр в последовательности изображений, 
# из которых состоит видео

    cap = open_video_capture(load_video_source())
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)

        print('Наведите камеру на шахматную доску под разными углами.')
        print("'s' — сохранить кадр, 'c' — калибровать, 'q' — выйти")


        session_dir = FRAMES_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        image_size = None
        read_failures = 0  # кол-во идущих подряд неудачных чтений кадра

        while True:
            ret, frame = cap.read()

            if not ret:
                read_failures += 1
                if read_failures >= MAX_READ_FAILURES:
                    print('Поток не отдаёт кадры, останавливаю захват')
                    break
                # Пауза перед следующей попыткой; waitKey заодно прокачивает очередь
                # событий окна, поэтому оно не «зависает», и даёт выйти по 'q'
                #if cv2.waitKey(READ_RETRY_DELAY_MS) & 0xFF == ord('q'):
                #    break
                continue

            read_failures = 0
            image_size = frame.shape[1::-1]  # (width, height)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, 
                                                       CHESSBOARD_SIZE, None)
            display = frame.copy()
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            criteria)
                cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, 
                                          found)

            status = f'Сохранено кадров: {len(object_points)}'
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)
            cv2.imshow('Camera calibration', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            elif key == ord("s"):
                if found:
                    object_points.append(OBJECT_POINTS_3D)
                    image_points.append(corners)
                    save_calibration_frame(session_dir, len(object_points), 
                                           frame, corners)
                    print(f"Кадр сохранён ({len(object_points)} всего)")
                else:
                    print("Доска не найдена на кадре — не сохранено")
            elif key == ord("c"):
                if len(object_points) < MIN_FRAMES_FOR_CALIBRATION:
                    print(f"Нужно минимум {MIN_FRAMES_FOR_CALIBRATION} кадров,"/ 
                          f" сейчас {len(object_points)}")
                    continue
                break
    finally: 
        cap.release()
        cv2.destroyAllWindows()

    if len(object_points) < MIN_FRAMES_FOR_CALIBRATION:
        print('Недостаточно кадров для калибровки')
        return

    print('Калибровка...')
    rms_error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)

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

    with open(f'{session_dir}/camera_calibration.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nRMS ошибка репроекции: {rms_error:.4f} (хорошо, если < 0.5-1.0 px)")
    print("Результат сохранён в camera_calibration.json")
    print(f"f_x = {fx:.2f}")
    print(f"f_y = {fy:.2f}")
    print(f"c_x = {cx:.2f}")
    print(f"c_y = {cy:.2f}")
    print("END")


if __name__ == "__main__":
    main()
