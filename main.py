import numpy as np
from PIL import Image
from l_u import solve_lin_system_with_lu
from scipy.linalg import solve
import cv2 as cv


def print_matrix(matrix, title=""):
    print(title)
    for line in matrix:
        print('     '.join(map(str, line)))
    print()


KEY_WIDTH = KEY_HEIGHT = INPUT_HEIGHT = 4
INPUT_WIDTH = 1
np.random.seed(0)
matrix_key = np.round(np.random.uniform(0, 10, (KEY_HEIGHT, KEY_WIDTH)), 1)
b = np.round(np.random.uniform(0, 10, (INPUT_HEIGHT, INPUT_WIDTH)), 1)


def encryption(key, input_data):
    return np.dot(key, input_data)


def decryption(key, input_data):
    return solve_lin_system_with_lu(key, input_data)


def decryption_2(key, input_data):
    return np.linalg.solve(key, input_data)


def decryption_3(key, input_data):
    return np.dot(np.linalg.inv(key), input_data)


def decryption_4(key, input_data):
    return solve(key, input_data)


def split(string, n):
    result = [string[index: index + n] for index in range(0, len(string), n)]
    while len(result[-1]) != n:
        result[-1] += " "
    return result


def str_to_ascii(string):
    return [ord(char) for char in string]


def str_to_int(string, n):
    return [[str_to_ascii(group) for group in element] for element in split(string, n)]


def int_to_str(ascii_str, code_word):
    ascii_str = np.round(np.reshape(ascii_str, (len(code_word), INPUT_HEIGHT)), 0).astype(int)
    return str.strip("".join(np.ndarray.flatten(np.array([[chr(i) for i in j] for j in ascii_str]))))


print_matrix(matrix_key, "A =")
print_matrix(b, "b =")

encrypted = encryption(matrix_key, b)
decrypted = decryption(matrix_key, encrypted)
decrypted_2 = decryption_2(matrix_key, encrypted)
decrypted_3 = decryption_3(matrix_key, encrypted)
decrypted_4 = decryption_4(matrix_key, encrypted)
print_matrix(encrypted, "encrypted floating point vector")
print_matrix(decrypted, "decrypted floating point vector with my own implementation of lu decomposition")
print_matrix(decrypted_2, "decrypted floating point vector with with np.linalg.solve()")
print_matrix(decrypted_3, "decrypted floating point vector with with np.dot()")
print_matrix(decrypted_4, "decrypted floating point vector with scipy.linalg.solve()")

word = "Aliens are watching us!"
print(word)
split_word = split(word, INPUT_HEIGHT)
print(split_word, end="\n\n")
code = np.array(str_to_int(split_word, INPUT_WIDTH))

encrypted_word = np.array([encryption(matrix_key, np.transpose(code[i])) for i in range(len(code))])
print_matrix(np.transpose(encrypted_word, (1, 0, 2)),
             "encrypted string (transposed for better visualisation)")
decrypted_word = np.array([decryption(matrix_key, encrypted_word[i]) for i in range(len(code))])
print_matrix(np.transpose(decrypted_word, (1, 0, 2)).astype(int),
             "decrypted string (transposed for better visualisation)")

starting_str = int_to_str(decrypted_word, code)
print(starting_str)


def img_encryption(key, img):
    n = key.shape[0]
    y = np.zeros((n, n, 3), dtype=np.double)

    for i in range(n):
        for j in range(n):
            y[i, j] = np.dot(key[i, :], img[:, j])
    return y


def img_decryption(key, img):
    n = key.shape[0]
    key = np.linalg.inv(key)
    y = np.zeros((n, n, 3), dtype=np.double)

    for i in range(n):
        for j in range(n):
            y[i, j] = np.dot(key[i, :], img[:, j])
    return y


width = height = 100
img_matrix_key = np.random.randint(0, 10, (width, height))

IMG = Image.open("image.jpg")
IMG = IMG.resize((width, height))

mat = np.array(IMG)
# Image.fromarray(mat, "RGB").show()
encrypted_img = img_encryption(img_matrix_key, mat)
decrypted_img = np.round(img_decryption(img_matrix_key, encrypted_img), 0).astype(int)

print((mat == decrypted_img).all())

''' 
    encryption of audio files are achievable too by dividing frequencies into groups of desired length and presenting
    them as a column vectors. after that you can multiply it by Key matrix to get encrypted data, decryption would be
    similar just with inverse of Key matrix
'''

'''
    It is possible to incorporate random numbers in encryption algorithm. Only key point is we should keep track of
    random number generator seed to be able to decrypt data. At first glace decryption is harder with random numbers is 
    key matrix, but if somebody gets a hand of seed then it becomes trivial to decrypt data.
    We could generate truly random numbers by observing physical processes, however keeping track of the data, that
    would help addressee encrypt message would become much more strenuous.
'''
