import cv2
import math
import numpy as np
import argparse

#BGR
MAX_WHITE = (255, 255, 255)
MIN_WHITE = (180, 180, 170)
MIN_YELLOW = (75, 120, 170)
MAX_YELLOW = (105, 180, 200)

NEIGHBOUR_MATCH_PERCENTAGE = 0.25
NEIGHBOUR_MATCH_WIDTH = 2

SLOPE_ANGLE_START = 20
SLOPE_ANGLE_STOP = 70

Y_TOLERANCE = 2

CLUSTER_SLOPE_DELTA = 0.05
CLUSTER_CONSTANT_DELTA = 5

paths = ['C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img1.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img2.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img3.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img4.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img6.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img7.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img8.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img9.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img10.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img11.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img13.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img14.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img15.bmp',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img16.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img17.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img18.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img19.jpeg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img20.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img21.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img22.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img23.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img24.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img25.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img26.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img27.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img28.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img29.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img30.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img31.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img32.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img33.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img34.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img35.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img36.JPG',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img37.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img38.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img39.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img40.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img41.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img42.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img43.jpg',
         'C:\\Users\\Mihai\\PycharmProjects\\VA test\\images\\img44.jpg'
         ]


def parse_args():
    parser = argparse.ArgumentParser(description='Horizon detection')

    parser.add_argument('-p', '--imagePath', type=str, required=True, help='Path to the image')
    parser.add_argument('-s', '--short', action='store_true')

    return parser.parse_args()


def is_in_color_range(color):
    return (MIN_WHITE[0] <= color[0] <= MAX_WHITE[0] and MIN_WHITE[1] <= color[1] <= MAX_WHITE[1]
            and MIN_WHITE[2] <= color[2] <= MAX_WHITE[2]) or (MIN_YELLOW[0] <= color[0] <= MAX_YELLOW[0]
            and MIN_YELLOW[1] <= color[1] <= MAX_YELLOW[1] and MIN_YELLOW[2] <= color[2] <= MAX_YELLOW[2])


def convert_to_yellow_and_white(img):
    height, width, _ = img.shape

    blank_image = np.zeros((height, width, 3), np.uint8)

    for row in range(height // 2, len(img)):
        for column in range(len(img[0])):
            if is_in_color_range(img[row][column]):
                blank_image[row][column] = (255, 255, 255)

    return blank_image


def check_neighbours(img, row, column):
    matched_neighbours = 0

    for i in range(row - NEIGHBOUR_MATCH_WIDTH, row + NEIGHBOUR_MATCH_WIDTH + 1):
        for j in range(column - NEIGHBOUR_MATCH_WIDTH, column + NEIGHBOUR_MATCH_WIDTH + 1):
            if img[i][j][0] == 255:
                matched_neighbours += 1

    # Scadem atunci cand face match pe el insusi
    matched_neighbours -= 1

    return ((2 * NEIGHBOUR_MATCH_WIDTH + 1) ** 2 - 1) * NEIGHBOUR_MATCH_PERCENTAGE <= matched_neighbours


def filter_by_neighbour(img):
    height, width, _ = img.shape

    for row in range(int(height / 2), len(img) - NEIGHBOUR_MATCH_WIDTH):
        for column in range(NEIGHBOUR_MATCH_WIDTH, len(img[0]) - NEIGHBOUR_MATCH_WIDTH):
            # Daca e pixel alb si are vecinii de aceeasi culoare intr-o masura
            if img[row][column][0] == 255 and not check_neighbours(img, row, column):
                img[row][column] = (0, 0, 0)


def find_points_for_line(img, slope, constant, y_tolerance):
    matched_points = 0
    height, width, _ = img.shape

    # y = 0
    x_start = int(-constant / slope)

    # Operim x cand iesim din poza pe verticala
    x_stop = int((height - constant) / slope)

    # Verificam ca prima valoare a lui X (cea pentru care Y = 0) sa fie pozitiva
    # La unele drepte pentru Y = 0 ar putea da X negativ, ceea ce nu e permis
    if x_start < 0:
        x_start = 0

    # Verificam sa nu iasa X din imagine, fiindca la pante mici nu ajunge Y sa fie prea mare
    if x_stop >= 2 * width / 3:
        x_stop = int(2 * width / 3)

    for x in range(x_start, x_stop):
        y = int(slope * x + constant)
        #print('X: ' + str(x_value))

        # Cautam daca macar un pixel din toleranta este alb
        for j in range(y - y_tolerance, y + y_tolerance + 1):
            #print('J: ' + str(j))
            if j >= 0 and j < height:
                # x matematic este coloana in matricea pozei
                # y matematic este linia in matricea pozei
                row = height - j - 1
                column = x

                if img[row][column][0] == 255:
                    matched_points += 1
                    break

    return matched_points


def get_constants(img, slope):
    height, width, _ = img.shape
    constants = list()

    # Constantele pentru deplasarea pe OY
    for i in range(0, int(height / 2), 2):
        constants.append(i)

    # Constantele pentru deplasarea pe OX
    # Ne trebuie pentru y = 0
    # Din y = mx + n, dar y = 0 -> n = -mx
    for x in range(0, int(2 * width / 3), 2):
        constant = -slope * x
        constants.append(constant)

    return constants


def find_lines(img, slope_angle_start, slope_angle_stop):
    results = list()

    for angle in range(slope_angle_start, slope_angle_stop + 1):
        slope = math.tan(angle * math.pi / 180)
        constants = get_constants(img, slope)

        for constant in constants:
            matched_points = find_points_for_line(img, slope, constant, Y_TOLERANCE)
            results.append([constant, slope, matched_points])

    return results


def draw_line(img, result, y_tolerance, thickness):
    # BGR
    color_1 = (0, 0, 255)
    color_2 = (0, 255, 0)
    color_3 = (255, 0, 0)
    color_4 = (153, 51, 255)
    color_5 = (255, 255, 51)
    colors = [color_1, color_2, color_3, color_4, color_5]

    height, width, _ = img.shape

    constant = result[0]
    slope = result[1]

    # y = 0
    x_start = int(-constant / slope)

    # Operim x cand iesim din poza pe verticala
    x_stop = int((height - constant) / slope)

    # Verificam ca prima valoare a lui X (cea pentru care Y = 0) sa fie pozitiva
    # La unele drepte pentru Y = 0 ar putea da X negativ, ceea ce nu e permis
    if x_start < 0:
        x_start = 0

    # Verificam sa nu iasa X din imagine, fiindca la pante mici nu ajunge Y sa fie prea mare
    if x_stop >= width:
        x_stop = width

    for x in range(x_start, x_stop):
        y = int(slope * x + constant)
        #print('X: ' + str(x_value))

        # Cautam daca macar un pixel din toleranta este alb
        for j in range(y - y_tolerance, y + y_tolerance + 1):
            #print('J: ' + str(j))
            if j >= 0 and j < height:
                row = height - j - 1
                column = x

                for k in range(thickness):
                    if column - k >= 0 and column - k < height:
                        img[row][column] = colors[0]
                break

    return img


def get_new_equation(img, constant, slope):
    height, width, _ = img.shape

    x1 = 0
    y1 = slope * x1 + constant

    x2 = 100
    y2 = slope * x2 + constant

    x1 = width - x1
    x2 = width - x2

    new_slope = (y2 - y1) / (x2 - x1)
    new_constant = y2 - slope * x2

    return new_constant, new_slope


def get_constants2(img, slope):
    height, width, _ = img.shape
    constants = list()

    # Mergem pana cand dreapta intersecteaza limita pozei la inaltimea height / 3
    const_lim = int(height / 3 - slope * width)

    for i in range(int(width / 3), const_lim, 2):
        constants.append(i)

    return constants


def find_points_for_line2(img, slope, constant, y_tolerance):
    matched_points = 0
    height, width, _ = img.shape

    x_start = int(width / 3)

    # Verificam sa nu iasa X din imagine, fiindca la pante mici nu ajunge Y sa fie prea mare
    x_stop = width

    for x in range(x_start, x_stop):
        y = int(slope * x + constant)

        # Cautam daca macar un pixel din toleranta este alb
        for j in range(y - y_tolerance, y + y_tolerance + 1):
            if j >= 0 and j < height:
                # x matematic este coloana in matricea pozei
                # y matematic este linia in matricea pozei
                row = height - j - 1
                column = x

                if img[row][column][0] == 255:
                    matched_points += 1
                    break

    return matched_points


def draw_line2(img, result, y_tolerance, thickness):
    # BGR
    color_1 = (0, 0, 255)
    color_2 = (0, 255, 0)
    color_3 = (255, 0, 0)
    color_4 = (153, 51, 255)
    color_5 = (255, 255, 51)
    colors = [color_1, color_2, color_3, color_4, color_5]

    height, width, _ = img.shape

    constant = result[0]
    slope = result[1]

    # y = 0
    x_start = int(width / 3)

    # Operim x cand iesim din poza pe verticala
    x_stop = width

    for x in range(x_start, x_stop):
        y = int(slope * x + constant)

        # Cautam daca macar un pixel din toleranta este alb
        for j in range(y - y_tolerance, y + y_tolerance + 1):

            if j >= 0 and j < height:
                row = height - j - 1
                column = x

                for k in range(thickness):
                    if column - k >= 0 and column - k < height:
                        img[row][column] = colors[0]
                break

    return img


def find_lines2(img, slope_angle_start, slope_angle_stop):
    results = list()

    for angle in range(slope_angle_start, slope_angle_stop + 1):
        slope = -math.tan(angle * math.pi / 180)
        constants = get_constants2(img, slope)

        for constant in constants:
            matched_points = find_points_for_line2(img, slope, constant, Y_TOLERANCE)
            results.append([constant, slope, matched_points])

    return results


def make_red_line(image, x, y, thickness):
    height, width, _ = image.shape
    row = height - int(y)

    for j in range(len(image[0])):
        column = j

        for i in range(thickness):
            if row + i < height:
                image[row + i][column][0] = image[row + i][column][1] = 0
                image[row + i][column][2] = 255


def pyramid(img):
    height, width, channels = img.shape
    blank_image = np.zeros((int(height / 2), int(width / 2), channels), np.uint8)
    blank_image_line = 0

    for row_index in range(0, len(img) - len(img) % 2, 2):   # index in linii
        blank_image_column = 0

        for column_index in range(0, len(img[0]) - len(img[0]) % 2, 2):   # index in coloana
            avg_red = (int(img[row_index][column_index][2]) + int(img[row_index + 1][column_index + 1][2]) +
                       int(img[row_index + 1][column_index][2]) + int(img[row_index][column_index + 1][2])) / 4

            avg_green = (int(img[row_index][column_index][1]) + int(img[row_index + 1][column_index + 1][1]) +
                         int(img[row_index + 1][column_index][1]) + int(img[row_index][column_index + 1][1])) / 4

            avg_blue = (int(img[row_index][column_index][0]) + int(img[row_index + 1][column_index + 1][0]) +
                        int(img[row_index + 1][column_index][0]) + int(img[row_index][column_index + 1][0])) / 4

            blank_image[blank_image_line][blank_image_column][0] = avg_blue
            blank_image[blank_image_line][blank_image_column][1] = avg_green
            blank_image[blank_image_line][blank_image_column][2] = avg_red

            blank_image_column += 1

        blank_image_line += 1

    return blank_image


def shrink_image(img):
    num_of_shrinks = 0
    height, width, _ = img.shape

    while height >= 400 or width >= 400:
        num_of_shrinks += 1
        img = pyramid(img)
        height, width, _ = img.shape

    return img, num_of_shrinks


def main():
    args = parse_args()
    path = args.imagePath
    short_version = args.short

    img = cv2.imread(path)
    num_of_shrinks = 0

    yellow_and_white = convert_to_yellow_and_white(img)
    filter_by_neighbour(yellow_and_white)

    if short_version:
        yellow_and_white, num_of_shrinks = shrink_image(yellow_and_white)

    # Gaseste liniile pe stanga
    results_left = find_lines(yellow_and_white, SLOPE_ANGLE_START, SLOPE_ANGLE_STOP)
    results_left.sort(key=lambda y: y[2])

    results_left = results_left[-5:]
    avg_slope_1 = (results_left[0][1] + results_left[1][1] + results_left[2][1] + results_left[3][1] + results_left[4][1]) / 5
    avg_constant_1 = (results_left[0][0] + results_left[1][0] + results_left[2][0] + results_left[3][0] + results_left[4][0]) / 5

    # Gaseste liniile pe dreapta
    results_left = find_lines2(yellow_and_white, SLOPE_ANGLE_START, SLOPE_ANGLE_STOP)
    results_left.sort(key=lambda y: y[2])

    results_left = results_left[-5:]
    avg_slope_2 = (results_left[0][1] + results_left[1][1] + results_left[2][1] + results_left[3][1] + results_left[4][1]) / 5
    avg_constant_2 = (results_left[0][0] + results_left[1][0] + results_left[2][0] + results_left[3][0] + results_left[4][0]) / 5

    # Gasim coordonatele itersectiei punctelor
    x = int((avg_constant_2 - avg_constant_1) / (avg_slope_1 - avg_slope_2))
    y = avg_slope_1 * x + avg_constant_1

    # Ajustam coordonatele dupa micsorare
    while num_of_shrinks:
        y *= 2
        x *= 2
        num_of_shrinks -= 1

    make_red_line(img, x, y, 5)
    cv2.imshow('Image', img)
    #print(i)
    #cv2.imwrite('C:\\Users\\Mihai\\PycharmProjects\\VA test\\output\\Image' + str(i) + '.jpg', img)
    #cv2.imshow('dd', yellow_and_white)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()