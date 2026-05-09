import cv2
import numpy as np

def count_and_draw_red_clusters(
    image_path,
    min_area=200,
    blur_kernel=5,
    draw=True
):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    output = img.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color ranges (two ranges in HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Blur to merge nearby regions
    if blur_kernel > 1:
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)

    # Binary cleanup
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    # Morphology (optional but recommended)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours (clusters)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cluster_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        cluster_count += 1

        if draw:
            # Draw contour
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

            # Draw center + label
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(output, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(
                    output,
                    f"{cluster_count}",
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

    return {
        "count": cluster_count,
        "mask": mask,
        "output": output
    }

def main():


    #blob_detection()
    result = count_and_draw_red_clusters(
        "image2.tiff",
        min_area=300
    )

    print("Clusters:", result["count"])

    cv2.imshow("Red clusters", result["output"])
    cv2.imshow("Mask", result["mask"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
