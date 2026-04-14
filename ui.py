import cv2
from datetime import datetime


def draw_ui(frame, present_list):
    height, width, _ = frame.shape

    # ================= HEADER ================= #
    header_color = (45, 45, 45)
    cv2.rectangle(frame, (0, 0), (width, 50), header_color, -1)

    cv2.putText(frame, "SMART FACE ATTENDANCE SYSTEM",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)

    # Time & Date
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%d-%m-%Y")

    cv2.putText(frame, f"{date_str} | {time_str}",
                (width - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    # ================= RIGHT PANEL ================= #
    panel_color = (30, 30, 30)
    cv2.rectangle(frame, (750, 50), (width, height), panel_color, -1)

    # Panel Title
    cv2.putText(frame, "PRESENT TODAY",
                (780, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    # Divider Line
    cv2.line(frame, (750, 90), (width, 90), (80, 80, 80), 1)

    # ================= PRESENT LIST ================= #
    y = 120
    for i, person in enumerate(present_list[-15:]):
        # Alternating row colors
        if i % 2 == 0:
            cv2.rectangle(frame, (750, y - 15), (width, y + 10), (40, 40, 40), -1)

        cv2.putText(frame, person,
                    (760, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        y += 25

    # ================= FOOTER ================= #
    footer_color = (45, 45, 45)
    cv2.rectangle(frame, (0, height - 40), (width, height), footer_color, -1)

    cv2.putText(frame, "Press 'Q' to Exit",
                (20, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    # Status Indicator
    cv2.circle(frame, (width - 30, height - 20), 8, (0, 255, 0), -1)
    cv2.putText(frame, "LIVE",
                (width - 90, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)

    return frame